"""Search from the command line.

    uv run python -m omaha.retrieve.search_cli --query "who is out with a foot injury"
    uv run python -m omaha.retrieve.search_cli --query "..." --as-of 2025-12-19T17:00:00Z

`--as-of` is the one worth playing with. Ask the same question with and without it and
the answers should differ, because Thursday's report and Saturday's are different rows
and the store never overwrote one with the other.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from omaha.db.session import session_scope
from omaha.retrieve.search import hybrid_search


def _embedder():
    try:
        from omaha.retrieve.embed import embed_query

        return embed_query
    except Exception:
        print("note: no embedding model here — lexical only\n", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omaha.retrieve.search_cli")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--as-of", default=None, help="ISO timestamp, e.g. 2025-12-19T17:00:00Z")
    args = parser.parse_args(argv)

    as_of = None
    if args.as_of:
        as_of = dt.datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))

    with session_scope() as session:
        hits = hybrid_search(
            session,
            args.query,
            limit=args.limit,
            candidates=args.candidates,
            as_of=as_of,
            embed_query_fn=_embedder(),
        )

        if not hits:
            print("no results")
            return 0

        if as_of:
            print(f"as of {as_of.isoformat()}\n")

        for position, hit in enumerate(hits, start=1):
            ranks = f"d={hit.dense_rank or '-'} l={hit.lexical_rank or '-'}"
            print(f"{position:>2}. [{hit.score:.4f}] {hit.found_by:<8} {ranks}")
            print(f"    {hit.text[:220]}")
            print(f"    knew at {hit.knowledge_time:%Y-%m-%d %H:%M}  {hit.source_url[:88]}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
