"""Retrieval evaluation CLI.

    uv run python -m omaha.eval.run --mode lexical          # runs on the host
    make eval                                               # hybrid, in the container

Compares retrieval strategies over the gold set. Run it before and after any change to
chunking, embedding or fusion — the point of having a number is to see it move.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omaha.db.session import session_scope
from omaha.eval.score import Report, evaluate_one
from omaha.retrieve.search import (
    dense_search,
    hybrid_search,
    lexical_search,
)

GOLD_PATH = Path("data/gold/injury_questions.json")


def load_gold(path: Path) -> list[dict]:
    if not path.exists():
        print(f"no gold set at {path}", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(path.read_text())["questions"]


def _embedder():
    """The embedding model, or None on a host where it isn't installed.

    Returning None rather than raising lets lexical evaluation run anywhere, which is
    what makes it usable as a fast check outside the container.
    """
    try:
        from omaha.retrieve.embed import embed_query

        return embed_query
    except Exception as exc:
        print(f"note: dense retrieval unavailable ({type(exc).__name__}); ", end="")
        print("run inside the worker container for hybrid mode\n")
        return None


def run_mode(session, questions: list[dict], mode: str, limit: int, embed_fn) -> Report:
    results = []

    for entry in questions:
        query = entry["question"]

        if mode == "lexical":
            hits = lexical_search(session, query, limit=limit)
        elif mode == "dense":
            if embed_fn is None:
                raise SystemExit("dense mode needs the embedding model")
            hits = dense_search(session, embed_fn(query), limit=limit)
        else:
            hits = hybrid_search(session, query, limit=limit, embed_query_fn=embed_fn)

        results.append(
            evaluate_one(
                entry["id"],
                query,
                entry["must_contain"],
                [h.text for h in hits[:limit]],
            )
        )

    return Report(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omaha.eval.run")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["lexical", "dense", "hybrid", "all"],
        help="'all' compares the three, which is the interesting output",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args(argv)

    questions = load_gold(args.gold)
    embed_fn = _embedder()

    modes = ["lexical", "dense", "hybrid"] if args.mode == "all" else [args.mode]
    if embed_fn is None:
        modes = [m for m in modes if m == "lexical"]

    reports: dict[str, Report] = {}
    with session_scope() as session:
        for mode in modes:
            reports[mode] = run_mode(session, questions, mode, args.limit, embed_fn)

    print(f"gold set: {len(questions)} questions, top-{args.limit}\n")
    header = f"{'mode':<10}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'hit@10':>8}{'MRR':>8}"
    print(header)
    print("-" * len(header))
    for mode, report in reports.items():
        print(
            f"{mode:<10}"
            f"{report.hit_rate_at(1):>7.0%}"
            f"{report.hit_rate_at(3):>8.0%}"
            f"{report.hit_rate_at(5):>8.0%}"
            f"{report.hit_rate_at(10):>8.0%}"
            f"{report.mrr:>8.3f}"
        )

    if args.show_misses:
        for mode, report in reports.items():
            if report.misses:
                print(f"\n{mode} missed:")
                for miss in report.misses:
                    print(f"  {miss.question_id:<26} {miss.question[:60]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
