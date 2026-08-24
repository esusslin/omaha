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
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from omaha.config import get_settings
from omaha.db.models import Document, JobRun, Source
from omaha.db.session import session_scope
from omaha.ingest import store
from omaha.ingest.fetch import fetch
from omaha.ingest.parse import parse
from omaha.ingest.sweep import sweep as run_sweep

settings = get_settings()


def ingest_source(session: Session, source: Source, client: httpx.Client | None = None) -> str:
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
        outcome = run_sweep(
            session, job_name="cli", kind=args.kind, name=args.name, due_only=not args.force
        )
    if not outcome.lines:
        print("nothing due")
        return 0
    for line in outcome.lines:
        print(line)
    print(f"-- attempted {outcome.attempted}, failed {outcome.failed}, created {outcome.created}")
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


def cmd_seed(args: argparse.Namespace) -> int:
    """Register all 32 clubs. Idempotent — existing sources are left alone.

    Also retires any legacy source pointing straight at a `/team/injury-report/` page,
    because that page's table is client-rendered: it fetches 200, parses to the legend,
    and reports healthy forever while storing nothing.
    """
    from omaha.ingest.seeds import all_seeds, unavailable_source_names

    added = skipped = retired = disabled = 0
    with session_scope() as session:
        for seed in all_seeds():
            existing = session.scalars(select(Source).where(Source.name == seed.name)).first()
            if existing:
                skipped += 1
                continue
            session.add(
                Source(
                    name=seed.name,
                    kind=seed.kind,
                    url=seed.url,
                    team=seed.team,
                    cadence_seconds=seed.cadence_seconds,
                )
            )
            added += 1

        if not args.keep_legacy:
            stale = session.scalars(
                select(Source).where(
                    Source.kind == "injury_report",
                    Source.url.like("%/team/injury-report%"),
                    Source.enabled.is_(True),
                )
            ).all()
            for source in stale:
                source.enabled = False
                source.last_error = "disabled: injury table is client-rendered, use the index"
                retired += 1

        # Sources registered before we learned the club doesn't serve them. Disabled
        # rather than deleted, and carrying the reason, so /health reports a known
        # exception instead of an unexplained absence — and so `consecutive_failures`
        # stops climbing on something nobody intends to fix.
        for name, reason in unavailable_source_names().items():
            registered = session.scalars(select(Source).where(Source.name == name)).first()
            if registered is not None and registered.enabled:
                registered.enabled = False
                registered.last_error = f"disabled: {reason}"
                disabled += 1

        session.flush()

    print(f"added {added}, already present {skipped}, retired {retired}, disabled {disabled}")
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    """Show what an index links to, without storing anything.

    Coverage varies by club — some list a full season of reports, some only latest news.
    This is how you find out which, rather than assuming.
    """
    import httpx

    from omaha.ingest.discover import discover_links

    with session_scope() as session:
        query = select(Source).where(Source.kind == "injury_index")
        if args.name:
            query = query.where(Source.name == args.name)
        sources = session.scalars(query.order_by(Source.name)).all()

        if not sources:
            print("no index sources registered — run `seed` first")
            return 1

        with httpx.Client(timeout=settings.request_timeout_seconds) as client:
            for source in sources:
                result = fetch(source.url, client=client)
                if not result.ok or not result.content:
                    print(f"{source.name:<22} FAIL {result.error or result.status_code}")
                    continue

                html = result.content.decode("utf-8", errors="replace")
                links = discover_links(html, base_url=result.url)
                unseen = sum(1 for x in links if not store.has_seen_url(session, source, x.url))
                print(f"{source.name:<22} {len(links):>3} linked, {unseen:>3} new")

                if args.verbose:
                    for link in links[: args.limit]:
                        print(f"    {link.title[:70]}")
    return 0


def cmd_reparse(args: argparse.Namespace) -> int:
    """Re-run the parsers over stored originals, refreshing `parsed_text`/`parsed_tables`.

    Extraction is frozen at ingest time: `parse()` runs once and its output is persisted.
    So improving a parser changes nothing for documents already collected — re-chunking
    just replays the stored tables. This is the step that closes that loop, and the
    reason `raw_ref` keeps every original byte.

    Documents stay immutable where it counts: `content_hash`, `knowledge_time` and the
    raw file are untouched. Only derived fields are rewritten. `text_hash` is refreshed
    too, since it describes the parsed text and would otherwise start lying.
    """
    from pathlib import Path

    from omaha.ingest.store import text_fingerprint

    with session_scope() as session:
        query = select(Document).order_by(Document.id)
        if args.doc_type:
            query = query.where(Document.doc_type == args.doc_type)
        documents = session.scalars(query.limit(args.limit)).all()

        changed = missing = 0
        for document in documents:
            if not document.raw_ref or not Path(document.raw_ref).exists():
                missing += 1
                continue

            content = Path(document.raw_ref).read_bytes()
            parsed = parse(content, content_type=None, url=document.source_url)
            if parsed.is_empty:
                continue

            # Count rows, not tables. The prose extractor always emits exactly one
            # table, so comparing table counts reports "unchanged" no matter how much
            # the extraction improved — which is worse than no metric at all.
            def _rows(tables: list[dict[str, Any]] | None) -> int:
                return sum(len(t.get("rows", [])) for t in (tables or []))

            before = (document.text_hash, _rows((document.parsed_tables or {}).get("tables")))
            document.parsed_text = parsed.text or None
            document.parsed_tables = (
                {"tables": parsed.tables, "parser": parsed.parser} if parsed.tables else None
            )
            document.text_hash = text_fingerprint(parsed.text)
            after = (document.text_hash, _rows(parsed.tables))

            if before != after:
                changed += 1
                if args.verbose:
                    print(f"doc {document.id}: rows {before[1]} -> {after[1]} ({parsed.parser})")

        session.flush()

    print(f"reparsed {len(documents)} documents, {changed} changed")
    if missing:
        print(f"  {missing} had no readable raw file — those cannot be reparsed")
    print("\nnow: uv run python -m omaha.retrieve.run chunk --rechunk")
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
    p_sweep.add_argument("--name", default=None, help="one source, for a cautious first run")
    p_sweep.add_argument("--force", action="store_true", help="ignore cadence and fetch everything")
    p_sweep.set_defaults(func=cmd_sweep)

    p_jobs = sub.add_parser("jobs", help="recent scheduled runs")
    p_jobs.add_argument("--limit", type=int, default=20)
    p_jobs.set_defaults(func=cmd_jobs)

    p_seed = sub.add_parser("seed", help="register all 32 clubs")
    p_seed.add_argument(
        "--keep-legacy",
        action="store_true",
        help="don't disable old client-rendered injury-report sources",
    )
    p_seed.set_defaults(func=cmd_seed)

    p_re = sub.add_parser("reparse", help="re-run parsers over stored originals")
    p_re.add_argument("--limit", type=int, default=1000)
    p_re.add_argument("--doc-type", default=None)
    p_re.add_argument("--verbose", action="store_true")
    p_re.set_defaults(func=cmd_reparse)

    p_disc = sub.add_parser("discover", help="show what indexes link to, storing nothing")
    p_disc.add_argument("--name", default=None, help="one index source; default is all")
    p_disc.add_argument("--verbose", action="store_true", help="print link titles")
    p_disc.add_argument("--limit", type=int, default=10)
    p_disc.set_defaults(func=cmd_discover)

    p_asof = sub.add_parser("asof", help="what did we know at a point in time?")
    p_asof.add_argument("--name", required=True)
    p_asof.add_argument("--when", required=True, help="ISO 8601, e.g. 2026-09-10T18:00:00Z")
    p_asof.set_defaults(func=cmd_asof)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
