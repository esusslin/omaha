"""Sweeping sources on a cadence, with job-run accounting.

Two things here that the Day 2 CLI didn't have:

**Cadence gating.** A source declares how often it wants to be polled. The scheduler
fires often; the sweep decides who is actually due. Without this, a 5-minute tick hits
every source every time — which is how `the-algo` burned 52,000 API credits a month
against a 20,000 budget before it grew throttle tiers.

**Job-run records.** Every sweep writes a row saying what it attempted and what it
found, so a missed Wednesday is visible rather than inferred.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from omaha.config import get_settings
from omaha.db.models import JobRun, Source
from omaha.ingest import store
from omaha.ingest.fetch import fetch
from omaha.ingest.parse import parse

settings = get_settings()


@dataclass
class SweepOutcome:
    job_name: str
    attempted: int = 0
    failed: int = 0
    created: int = 0
    lines: list[str] | None = None

    def __post_init__(self) -> None:
        if self.lines is None:
            self.lines = []


def is_due(source: Source, now: dt.datetime) -> bool:
    """Has enough time passed since the last attempt for this source's cadence?

    Uses last *attempt*, not last success — otherwise a failing source gets retried on
    every tick and hammers an origin that's already unhappy.
    """
    if not source.enabled:
        return False
    if source.last_attempt_at is None:
        return True
    elapsed = (now - source.last_attempt_at).total_seconds()
    return elapsed >= source.cadence_seconds


def ingest_one(
    session: Session, source: Source, client: httpx.Client | None = None
) -> tuple[str, bool, bool]:
    """Fetch/parse/store one source.

    Returns (status line, failed, created_document).
    """
    result = fetch(source.url, etag=source.etag, last_modified=source.last_modified, client=client)

    if result.error or result.status_code not in (200, 304):
        store.record_failure(
            session, source, result.error or f"HTTP {result.status_code}", result.fetched_at
        )
        return f"FAIL   {source.name}: {result.error or result.status_code}", True, False

    if result.not_modified:
        store.record_unchanged(session, source, result.fetched_at)
        return f"304    {source.name}: unchanged", False, False

    parsed = parse(result.content or b"", content_type=result.content_type, url=result.url)
    if parsed.is_empty:
        store.record_failure(session, source, "parsed to empty text", result.fetched_at)
        return f"EMPTY  {source.name}: nothing extracted ({parsed.parser})", True, False

    document = store.store_document(session, source, result, parsed)
    if document is None:
        return f"SAME   {source.name}: identical content, no new row", False, False

    return (
        f"NEW    {source.name}: doc {document.id}, "
        f"{len(parsed.text)} chars, {len(parsed.tables)} tables",
        False,
        True,
    )


def sweep(
    session: Session,
    *,
    job_name: str = "manual",
    kind: str | None = None,
    due_only: bool = True,
) -> SweepOutcome:
    """Run one sweep, recording a JobRun.

    `due_only=False` forces every source regardless of cadence — useful from the CLI,
    never from the scheduler.
    """
    now = dt.datetime.now(dt.UTC)
    run = JobRun(job_name=job_name, started_at=now)
    session.add(run)
    session.flush()

    outcome = SweepOutcome(job_name=job_name)

    try:
        query = select(Source).where(Source.enabled.is_(True))
        if kind:
            query = query.where(Source.kind == kind)
        sources = session.scalars(query.order_by(Source.name)).all()

        candidates = [s for s in sources if not due_only or is_due(s, now)]

        with httpx.Client(timeout=settings.request_timeout_seconds) as client:
            for source in candidates:
                line, failed, created = ingest_one(session, source, client=client)
                outcome.attempted += 1
                outcome.failed += int(failed)
                outcome.created += int(created)
                assert outcome.lines is not None
                outcome.lines.append(line)

        run.ok = True
    except Exception as exc:
        run.ok = False
        run.error = repr(exc)
        raise
    finally:
        run.finished_at = dt.datetime.now(dt.UTC)
        run.sources_attempted = outcome.attempted
        run.sources_failed = outcome.failed
        run.documents_created = outcome.created
        session.flush()

    return outcome
