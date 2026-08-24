"""Extraction CLI.

    uv run python -m omaha.extract.run status          # how much is pending, and cost
    uv run python -m omaha.extract.run extract --limit 20
    uv run python -m omaha.extract.run show --team PHI
    uv run python -m omaha.extract.run coverage        # how much of the corpus yielded facts

`status` before `extract`, always. It prints the estimated spend for the run you're
about to start, because "how many chunks is that" is a question worth answering before
the API call rather than after the invoice.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import func, select

from omaha.db.models import Chunk, InjuryRecord
from omaha.db.session import session_scope
from omaha.extract import client, store
from omaha.extract.prompt import EXTRACTOR_VERSION

# Rough per-chunk token counts, used only for the pre-flight estimate. Haiku 4.5 is
# $1/M input and $5/M output; being wrong by a factor of two here still gives the right
# order of magnitude, which is all this needs to do.
EST_INPUT_TOKENS = 600
EST_OUTPUT_TOKENS = 200
INPUT_COST_PER_M = 1.0
OUTPUT_COST_PER_M = 5.0


def estimate_cost(chunks: int) -> float:
    return (
        chunks * EST_INPUT_TOKENS / 1e6 * INPUT_COST_PER_M
        + chunks * EST_OUTPUT_TOKENS / 1e6 * OUTPUT_COST_PER_M
    )


def cmd_status(_: argparse.Namespace) -> int:
    with session_scope() as session:
        total = session.scalar(select(func.count(Chunk.id))) or 0
        pending = store.pending_count(session, EXTRACTOR_VERSION)
        records = (
            session.scalar(
                select(func.count(InjuryRecord.id)).where(
                    InjuryRecord.extractor_version == EXTRACTOR_VERSION
                )
            )
            or 0
        )

    print(f"extractor        {EXTRACTOR_VERSION}")
    print(f"api key          {'present' if client.available() else 'MISSING'}")
    print(f"chunks           {total}")
    print(f"pending          {pending}")
    print(f"records          {records}")
    print(f"est. cost        ${estimate_cost(pending):.2f} to clear the backlog")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    if not client.available():
        print("ANTHROPIC_API_KEY is not set — nothing to do", file=sys.stderr)
        return 1

    processed = written = failed = 0
    remaining = args.limit

    # **Checkpoint every `--batch` chunks.** The first version of this held one
    # transaction open for the whole run, so a 1,173-chunk backfill was all-or-nothing:
    # any interruption rolled back every record while the API calls stayed paid for.
    # Committing in batches means a failure costs one batch, and the stamped chunks mean
    # the next invocation picks up where this one stopped.
    while remaining > 0:
        size = min(args.batch, remaining)
        with session_scope() as session:
            chunks = store.pending_chunks(session, EXTRACTOR_VERSION, limit=size)
            if not chunks:
                break

            for chunk in chunks:
                document = chunk.document
                team = document.team if document else None
                # `published_time`, not `knowledge_time`: the model is resolving "today"
                # as the article's author meant it, which is the publication date. When
                # we happened to fetch it is irrelevant to what day the prose describes.
                published = document.published_time if document else None
                try:
                    drafts = client.extract(chunk.text, team_hint=team, published=published)
                except Exception as exc:
                    # One bad chunk shouldn't end the run. It stays unstamped, so the
                    # next pass retries it.
                    failed += 1
                    print(f"  chunk {chunk.id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue

                count = store.persist(session, chunk, drafts, version=EXTRACTOR_VERSION)
                processed += 1
                written += count
                if args.verbose and count:
                    print(f"  chunk {chunk.id}: {count} records")

            session.flush()

        remaining -= len(chunks)
        # Printed after the commit, so the number reported is the number that's durable.
        # A progress line that runs ahead of the transaction is a lie you'll believe.
        print(f"  ...{processed} processed, {written} records, {failed} failed", flush=True)

        if failed and processed == 0:
            print("every chunk in the first batch failed — stopping", file=sys.stderr)
            break

    print(f"\nprocessed {processed}, records {written}, failed {failed}")
    print(f"est. spend ${estimate_cost(processed):.3f}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    with session_scope() as session:
        statement = select(InjuryRecord).where(InjuryRecord.extractor_version == EXTRACTOR_VERSION)
        if args.team:
            statement = statement.where(InjuryRecord.team == args.team.upper())
        if args.player:
            statement = statement.where(InjuryRecord.player_name.ilike(f"%{args.player}%"))
        records = session.scalars(
            statement.order_by(InjuryRecord.knowledge_time.desc()).limit(args.limit)
        ).all()

        if not records:
            print("no records")
            return 0

        for record in records:
            day = record.report_day or "?"
            print(
                f"{record.team or '??':<4} {record.player_name:<24} "
                f"{record.position or '--':<5} {day:<4} "
                f"practice={record.practice_status or '-':<8} "
                f"status={record.game_status or '-':<13} {record.injury or ''}"
            )
            if args.evidence and record.evidence:
                print(f"      > {record.evidence[:110]}")
    return 0


def cmd_coverage(_: argparse.Namespace) -> int:
    """What fraction of processed chunks produced facts, and how complete are they?

    The number that matters is *field* coverage, not row count. A thousand records with
    null practice status would look like success and be worth nothing to the model.
    """
    with session_scope() as session:
        processed = (
            session.scalar(
                select(func.count(Chunk.id)).where(Chunk.extracted_version == EXTRACTOR_VERSION)
            )
            or 0
        )
        productive = (
            session.scalar(
                select(func.count(func.distinct(InjuryRecord.chunk_id))).where(
                    InjuryRecord.extractor_version == EXTRACTOR_VERSION
                )
            )
            or 0
        )
        total = (
            session.scalar(
                select(func.count(InjuryRecord.id)).where(
                    InjuryRecord.extractor_version == EXTRACTOR_VERSION
                )
            )
            or 0
        )

        def filled(column) -> int:  # type: ignore[no-untyped-def]
            return (
                session.scalar(
                    select(func.count(InjuryRecord.id)).where(
                        InjuryRecord.extractor_version == EXTRACTOR_VERSION,
                        column.is_not(None),
                    )
                )
                or 0
            )

        share = f" ({productive / processed:.0%})" if processed else ""
        print(f"chunks processed     {processed}")
        print(f"  produced records   {productive}{share}")
        print(f"records              {total}\n")
        if total:
            for label, column in (
                ("team", InjuryRecord.team),
                ("position", InjuryRecord.position),
                ("injury", InjuryRecord.injury),
                ("practice_status", InjuryRecord.practice_status),
                ("game_status", InjuryRecord.game_status),
                ("report_day", InjuryRecord.report_day),
                ("evidence", InjuryRecord.evidence),
            ):
                n = filled(column)
                print(f"  {label:<18} {n:>5} ({n / total:.0%})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omaha.extract.run")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="pending count and estimated cost").set_defaults(func=cmd_status)

    p_ex = sub.add_parser("extract", help="extract records from pending chunks")
    p_ex.add_argument("--limit", type=int, default=25, help="total chunks this run")
    p_ex.add_argument(
        "--batch",
        type=int,
        default=25,
        help="commit every N chunks, so an interruption costs a batch and not the run",
    )
    p_ex.add_argument("--verbose", action="store_true")
    p_ex.set_defaults(func=cmd_extract)

    p_show = sub.add_parser("show", help="print extracted records")
    p_show.add_argument("--team", default=None)
    p_show.add_argument("--player", default=None)
    p_show.add_argument("--limit", type=int, default=30)
    p_show.add_argument("--evidence", action="store_true", help="print supporting text")
    p_show.set_defaults(func=cmd_show)

    sub.add_parser("coverage", help="field completeness").set_defaults(func=cmd_coverage)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
