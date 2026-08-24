"""Scheduled collection.

**Timezone matters here.** NFL policy requires practice reports by 4:00 pm Eastern on
Wednesday, Thursday and Friday. So the injury sweep is cron'd in `America/New_York`,
not UTC — otherwise it drifts an hour twice a year and misses the window in November,
which is exactly when the reports start mattering.

We poll at 17:00 ET, an hour after the deadline, so late filers are captured.

Everything is idempotent: the sweep gates on each source's cadence, and the store only
writes when content actually changed. A double-fire costs a conditional request.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from omaha.db.session import session_scope
from omaha.ingest.sweep import sweep

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")


def _run(job_name: str, kind: str | Sequence[str] | None, due_only: bool = True) -> None:
    """Run a sweep in its own session, logging rather than raising.

    A scheduler job that raises kills nothing but its own next run — but it also loses
    the outcome, so we catch, log, and let the JobRun row carry the detail.

    `due_only=False` exists for deadline-driven jobs. Cadence gating asks "has enough
    time passed since the last attempt?", which is right for a polling loop and wrong
    for a job that fires *because something just happened*. The 17:00 ET injury sweep
    was gated at 6 hours, so an hourly sweep at 16:07 made it skip everything and write
    a successful JobRun — a job that exists precisely to catch the 4pm filing deadline,
    quietly doing nothing on the three days of the week that matter.
    """
    try:
        with session_scope() as session:
            outcome = sweep(session, job_name=job_name, kind=kind, due_only=due_only)
        logger.info(
            "sweep complete job=%s attempted=%d failed=%d created=%d",
            job_name,
            outcome.attempted,
            outcome.failed,
            outcome.created,
        )
        for line in outcome.lines or []:
            logger.info("  %s", line)
    except Exception:
        logger.exception("sweep failed job=%s", job_name)


def _index() -> None:
    """Chunk anything new, then embed it if the model is present.

    Without this the collector is only half a pipeline: documents land in the store and
    stay invisible to search, because chunking and embedding were manual `make` targets.
    Ingestion that doesn't reach the index is ingestion nobody can query.

    Embedding is skipped where `fastembed` isn't installed — the host — so the same
    scheduler runs on a laptop and in the Linux container, doing as much as it can in
    each. Both steps are idempotent and resumable, so a partial run costs nothing.
    """
    from sqlalchemy import select

    from omaha.db.models import Chunk, Document
    from omaha.retrieve.chunk import chunk_document

    try:
        with session_scope() as session:
            chunked = select(Chunk.document_id).distinct().scalar_subquery()
            documents = session.scalars(
                select(Document).where(Document.id.not_in(chunked)).limit(500)
            ).all()

            made = 0
            for document in documents:
                tables = (document.parsed_tables or {}).get("tables")
                for draft in chunk_document(
                    doc_type=document.doc_type,
                    text=document.parsed_text or "",
                    tables=tables,
                ):
                    session.add(
                        Chunk(
                            document_id=document.id,
                            ordinal=draft.ordinal,
                            text=draft.text,
                            section_path=draft.section_path,
                            span_start=draft.span_start,
                            span_end=draft.span_end,
                        )
                    )
                    made += 1
            if made:
                logger.info("indexed %d chunks from %d documents", made, len(documents))
    except Exception:
        logger.exception("chunking failed")
        return

    try:
        from omaha.retrieve.embed import embed_all
    except Exception as exc:
        logger.debug("no embedding model (%s) — chunks left unembedded", type(exc).__name__)
        return

    try:
        with session_scope() as session:
            count = embed_all(session)
        if count:
            logger.info("embedded %d chunks", count)
    except Exception:
        logger.exception("embedding failed")

    _extract()


def _extract() -> None:
    """Turn newly chunked text into typed records.

    The third step of the pipeline, and it was missing: `_index` chunked and embedded and
    then stopped, so Phase 4 ran only where someone typed the CLI command. Search worked
    in production and `/injuries` returned an empty list — a system half-deployed in
    exactly the way that looks fine from the outside.

    **Bounded per run.** Each pass extracts at most `extract_batch_size` chunks, so a
    backfill can't monopolise the hourly slot or spend the month's budget in one go. The
    remainder stays pending and the next run continues — stamped chunks make it
    resumable.

    Skipped silently when no API key is configured, because a laptop without one should
    still run the rest of the pipeline rather than erroring every hour.
    """
    from omaha.config import get_settings

    settings = get_settings()

    try:
        from omaha.extract import client, store
        from omaha.extract.prompt import EXTRACTOR_VERSION
    except Exception:
        logger.exception("extraction unavailable")
        return

    if not client.available():
        logger.debug("no ANTHROPIC_API_KEY — chunks left unextracted")
        return

    written = processed = failed = 0
    try:
        with session_scope() as session:
            chunks = store.pending_chunks(
                session, EXTRACTOR_VERSION, limit=settings.extract_batch_size
            )
            if not chunks:
                return
            for chunk in chunks:
                document = chunk.document
                try:
                    drafts = client.extract(
                        chunk.text,
                        team_hint=document.team if document else None,
                        published=document.published_time if document else None,
                    )
                except Exception as exc:
                    # One bad chunk costs one chunk. It stays unstamped and is retried.
                    failed += 1
                    logger.warning("extract failed chunk=%s %s", chunk.id, exc)
                    continue
                written += store.persist(session, chunk, drafts, version=EXTRACTOR_VERSION)
                processed += 1
    except Exception:
        logger.exception("extraction run failed")
        return

    if processed:
        logger.info("extracted %d records from %d chunks (%d failed)", written, processed, failed)


def build_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=EASTERN)

    # Practice reports: Wed/Thu/Fri at 17:00 ET, an hour after the league deadline.
    # Both kinds: `injury_index` sources discover article URLs, `injury_report` covers
    # any direct source still registered.
    scheduler.add_job(
        _run,
        CronTrigger(day_of_week="wed,thu,fri", hour=17, minute=0, timezone=EASTERN),
        # due_only=False: fire regardless of cadence. This is the deadline job.
        args=["injury_sweep", ["injury_index", "injury_report"], False],
        id="injury_sweep",
        max_instances=1,
        coalesce=True,  # a missed fire runs once, not N times
        misfire_grace_time=3600,
    )

    # Close the loop: fetched documents are useless until they're chunked and embedded.
    # Offset from the sweeps so it runs against what they just wrote.
    scheduler.add_job(
        _index,
        CronTrigger(minute=25, timezone=EASTERN),
        id="index",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=1800,
    )

    # Transactions and news: hourly. Cadence gating inside the sweep does the throttling.
    scheduler.add_job(
        _run,
        CronTrigger(minute=7, timezone=EASTERN),
        args=["hourly_sweep", None],
        id="hourly_sweep",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=1800,
    )

    return scheduler
