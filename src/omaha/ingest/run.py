"""Ingestion CLI.

    uv run python -m omaha.ingest.run add --name ne-injury --kind injury_report \
        --team NE --url https://www.patriots.com/team/injury-report/
    uv run python -m omaha.ingest.run once --name ne-injury
    uv run python -m omaha.ingest.run sweep
    uv run python -m omaha.ingest.run list
    uv run python -m omaha.ingest.run asof --name ne-injury --when 2026-09-10T18:00:00Z
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import httpx
from sqlalchemy import select

from omaha.config import get_settings
from omaha.db.models import Document, JobRun, Source
from omaha.db.session import session_scope
from omaha.ingest import store
from omaha.ingest.fetch import fetch
from omaha.ingest.parse import parse
from omaha.ingest.sweep import sweep as run_sweep

settings = get_settings()


def ingest_source(session, source: Source, client: httpx.Client | None = None) -> str:
    """Fetch, parse and store one source. Returns a one-line status for the console."""
    result = fetch(source.url, etag=source.etag, last_modified=source.last_modified, client=client)

    if result.error or result.status_code not in (200, 304):
        store.record_failure(
            session, source, result.error or f"HTTP {result.status_code}", result.fetched_at
        )
        return f"FAIL   {source.name}: {result.error or result.status_code}"

    if result.not_modified:
        store.record_unchanged(session, source, result.fetched_at)
        return f"304    {source.name}: unchanged"

    parsed = parse(result.content or b"", content_type=result.content_type, url=result.url)
    if parsed.is_empty:
        store.record_failure(session, source, "parsed to empty text", result.fetched_at)
        return f"EMPTY  {source.name}: parsed to nothing ({parsed.parser})"

    document = store.store_document(session, source, result, parsed)
    if document is None:
        return f"SAME   {source.name}: identical content, no new row"

    return (
        f"NEW    {source.name}: doc {document.id}, "
        f"{len(parsed.text)} chars, {len(parsed.tables)} tables, {parsed.parser}"
    )


def cmd_add(args: argparse.Namespace) -> int:
    with session_scope() as session:
        existing = session.scalars(select(Source).where(Source.name == args.name)).first()
        if existing:
            print(f"source '{args.name}' already exists (id {existing.id})")
            return 1
        source = Source(
            name=args.name,
            kind=args.kind,
            url=args.url,
            team=args.team,
            cadence_seconds=args.cadence,
        )
        session.add(source)
        session.flush()
        print(f"added source '{source.name}' (id {source.id})")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    with session_scope() as session:
        sources = session.scalars(select(Source).order_by(Source.name)).all()
        if not sources:
            print("no sources registered")
            return 0
        for s in sources:
            docs = session.scalar(select(Document.id).where(Document.source_id == s.id).limit(1))
            state = "stale" if s.is_stale else "ok"
            last = s.last_success_at.isoformat() if s.last_success_at else "never"
            print(
                f"{s.id:>3}  {s.name:<24} {s.kind:<16} {state:<6} "
                f"last_success={last}  docs={'yes' if docs else 'no'}"
            )
    return 0


def cmd_once(args: argparse.Namespace) -> int:
    with session_scope() as session:
        source = session.scalars(select(Source).where(Source.name == args.name)).first()
        if source is None:
            print(f"no source named '{args.name}'")
            return 1
        print(ingest_source(session, source))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Cadence-aware by default; --force ignores cadence (never do this on a schedule)."""
    with session_scope() as session:
        outcome = run_sweep(session, job_name="cli", kind=args.kind, due_only=not args.force)
    if not outcome.lines:
        print("nothing due")
        return 0
    for line in outcome.lines:
        print(line)
    print(
        f"-- attempted {outcome.attempted}, failed {outcome.failed}, " f"created {outcome.created}"
    )
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    """Recent scheduled runs."""
    from sqlalchemy import desc

    with session_scope() as session:
        runs = session.scalars(
            select(JobRun).order_by(desc(JobRun.started_at)).limit(args.limit)
        ).all()
        if not runs:
            print("no job runs yet")
            return 0
        for r in runs:
            dur = f"{r.duration_seconds:.1f}s" if r.duration_seconds is not None else "--"
            state = "ok" if r.ok else "FAILED"
            print(
                f"{r.id:>4}  {r.job_name:<16} {r.started_at.isoformat()}  {state:<7} "
                f"attempted={r.sources_attempted} failed={r.sources_failed} "
                f"created={r.documents_created} took={dur}"
            )
            if r.error:
                print(f"      error: {r.error}")
    return 0


def cmd_asof(args: argparse.Namespace) -> int:
    """Prove the bitemporal property from the command line."""
    when = dt.datetime.fromisoformat(args.when.replace("Z", "+00:00"))
    with session_scope() as session:
        source = session.scalars(select(Source).where(Source.name == args.name)).first()
        if source is None:
            print(f"no source named '{args.name}'")
            return 1
        document = store.as_of(session, source, when)
        if document is None:
            print(f"we knew nothing from '{args.name}' at {when.isoformat()}")
            return 0
        print(f"as of {when.isoformat()} we knew document {document.id}")
        print(f"  learned at : {document.knowledge_time.isoformat()}")
        print(f"  hash       : {document.content_hash[:12]}")
        print(f"  raw        : {document.raw_ref}")
        preview = (document.parsed_text or "")[:400]
        print(f"  text       : {preview}{'...' if len(document.parsed_text or '') > 400 else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omaha.ingest.run")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="register a source")
    p_add.add_argument("--name", required=True)
    p_add.add_argument(
        "--kind", required=True, help="injury_report | inactives | transactions | ..."
    )
    p_add.add_argument("--url", required=True)
    p_add.add_argument("--team", default=None)
    p_add.add_argument("--cadence", type=int, default=3600)
    p_add.set_defaults(func=cmd_add)

    sub.add_parser("list", help="show sources and health").set_defaults(func=cmd_list)

    p_once = sub.add_parser("once", help="ingest one source now")
    p_once.add_argument("--name", required=True)
    p_once.set_defaults(func=cmd_once)

    p_sweep = sub.add_parser("sweep", help="ingest sources that are due")
    p_sweep.add_argument("--kind", default=None)
    p_sweep.add_argument("--force", action="store_true", help="ignore cadence and fetch everything")
    p_sweep.set_defaults(func=cmd_sweep)

    p_jobs = sub.add_parser("jobs", help="recent scheduled runs")
    p_jobs.add_argument("--limit", type=int, default=20)
    p_jobs.set_defaults(func=cmd_jobs)

    p_asof = sub.add_parser("asof", help="what did we know at a point in time?")
    p_asof.add_argument("--name", required=True)
    p_asof.add_argument("--when", required=True, help="ISO 8601, e.g. 2026-09-10T18:00:00Z")
    p_asof.set_defaults(func=cmd_asof)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
