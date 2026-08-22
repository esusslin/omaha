"""Retrieval metrics.

Three numbers, each answering a different question:

- **hit rate @ k** — did the answer appear at all? The floor. If this is low nothing
  downstream can work, because generation cannot cite what retrieval never returned.
- **MRR** — how far down was it? Distinguishes "found it first" from "found it ninth",
  which hit rate flattens and which matters a great deal when a reranker or an LLM sees
  only the top few.
- **recall @ k** — of all the chunks that should have matched, how many came back?
  A player's status may appear in several documents; finding one is enough to answer,
  finding all is what you want before claiming coverage.

Relevance is a content predicate, not a chunk id. Chunk ids change every time the
extractor improves — and it improved four times this week — so a gold set pinned to them
would have needed rewriting each time and would quietly have measured nothing in between.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryResult:
    """One question's outcome."""

    question_id: str
    question: str
    relevant_positions: list[int]
    """1-based ranks of the hits that matched, in rank order."""
    total_relevant_found: int
    returned: int

    @property
    def hit(self) -> bool:
        return bool(self.relevant_positions)

    @property
    def first_position(self) -> int | None:
        return self.relevant_positions[0] if self.relevant_positions else None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_position if self.first_position else 0.0


@dataclass
class Report:
    results: list[QueryResult]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def hit_rate(self) -> float:
        return sum(r.hit for r in self.results) / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return sum(r.reciprocal_rank for r in self.results) / self.n if self.n else 0.0

    def hit_rate_at(self, k: int) -> float:
        if not self.n:
            return 0.0
        return sum(1 for r in self.results if r.first_position and r.first_position <= k) / self.n

    @property
    def misses(self) -> list[QueryResult]:
        """The questions worth reading. An aggregate hides which ones failed and why."""
        return [r for r in self.results if not r.hit]

    def summary(self, ks: tuple[int, ...] = (1, 3, 5, 10)) -> str:
        lines = [f"questions        {self.n}"]
        for k in ks:
            lines.append(f"  hit rate @{k:<3}     {self.hit_rate_at(k):.1%}")
        lines.append(f"  MRR              {self.mrr:.3f}")
        lines.append(f"  missed           {len(self.misses)}")
        return "\n".join(lines)


def is_relevant(text: str, must_contain: list[str]) -> bool:
    """Content predicate. Case-insensitive, all terms required.

    Deliberately strict about *which* row: "Cameron Latu" alone would match Tuesday,
    Wednesday and Thursday, so a question about Thursday names the day too. Retrieval
    that returns the right player on the wrong day is wrong, and a looser predicate
    would score it as correct.
    """
    haystack = text.casefold()
    return all(term.casefold() in haystack for term in must_contain)


def evaluate_one(
    question_id: str,
    question: str,
    must_contain: list[str],
    hit_texts: list[str],
) -> QueryResult:
    positions = [
        position
        for position, text in enumerate(hit_texts, start=1)
        if is_relevant(text, must_contain)
    ]
    return QueryResult(
        question_id=question_id,
        question=question,
        relevant_positions=positions,
        total_relevant_found=len(positions),
        returned=len(hit_texts),
    )
