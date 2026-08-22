"""Fusion and scoring. Pure functions — no database, no model.

The SQL paths need a live Postgres and are exercised by `make eval`; what's tested here
is the ranking logic, which is where a subtle bug would be invisible in a demo and fatal
in an evaluation.
"""

from __future__ import annotations

import datetime as dt

import pytest

from omaha.eval.score import Report, evaluate_one, is_relevant
from omaha.retrieve.search import RRF_K, SearchHit, reciprocal_rank_fusion

NOW = dt.datetime(2025, 12, 18, 21, 15, tzinfo=dt.UTC)


def hit(chunk_id: int, text: str = "") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=1,
        text=text or f"chunk {chunk_id}",
        section_path=None,
        source_url="https://example.com/report",
        doc_type="injury_report",
        team="PHI",
        knowledge_time=NOW,
        published_time=None,
    )


# --- RRF ---------------------------------------------------------------------------


def test_agreement_beats_single_list_dominance() -> None:
    """A chunk both retrievers like should outrank one that only one of them loves.
    This is the entire reason for fusing rather than concatenating."""
    dense = [hit(1), hit(2), hit(3)]
    lexical = [hit(9), hit(2), hit(8)]

    merged = reciprocal_rank_fusion({"dense": dense, "lexical": lexical})
    assert merged[0].chunk_id == 2


def test_score_is_sum_of_reciprocal_ranks() -> None:
    merged = reciprocal_rank_fusion({"dense": [hit(1)], "lexical": [hit(1)]})
    assert merged[0].score == pytest.approx(2 * (1 / (RRF_K + 1)))


def test_single_list_still_ranks_in_order() -> None:
    merged = reciprocal_rank_fusion({"lexical": [hit(5), hit(6), hit(7)]})
    assert [h.chunk_id for h in merged] == [5, 6, 7]


def test_records_where_each_hit_came_from() -> None:
    merged = reciprocal_rank_fusion({"dense": [hit(1), hit(2)], "lexical": [hit(2)]})
    by_id = {h.chunk_id: h for h in merged}

    assert by_id[2].found_by == "both"
    assert by_id[2].dense_rank == 2
    assert by_id[2].lexical_rank == 1
    assert by_id[1].found_by == "dense"
    assert by_id[1].lexical_rank is None


def test_deduplicates_across_lists() -> None:
    merged = reciprocal_rank_fusion({"dense": [hit(1), hit(2)], "lexical": [hit(2), hit(1)]})
    assert len(merged) == 2


def test_limit_is_applied_after_fusion() -> None:
    """Truncating before fusing would discard chunks that only rank once but rank well."""
    dense = [hit(i) for i in range(1, 11)]
    lexical = [hit(i) for i in range(11, 21)]
    assert len(reciprocal_rank_fusion({"dense": dense, "lexical": lexical}, limit=5)) == 5


def test_empty_input_is_not_an_error() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"dense": [], "lexical": []}) == []


def test_ties_break_deterministically() -> None:
    """Equal scores must not reorder between runs, or the eval numbers wobble."""
    a = reciprocal_rank_fusion({"dense": [hit(3), hit(1)], "lexical": [hit(1), hit(3)]})
    b = reciprocal_rank_fusion({"dense": [hit(3), hit(1)], "lexical": [hit(1), hit(3)]})
    assert [h.chunk_id for h in a] == [h.chunk_id for h in b]


# --- relevance predicate -----------------------------------------------------------

ROW = "Team: Eagles | Day: Thursday | Pos: TE | Player: Cameron Latu | Injury: Stinger | Practice: FULL"


def test_relevance_requires_every_term() -> None:
    assert is_relevant(ROW, ["Cameron Latu", "Thursday", "FULL"])
    assert not is_relevant(ROW, ["Cameron Latu", "Wednesday"])


def test_relevance_is_case_insensitive() -> None:
    assert is_relevant(ROW, ["cameron latu", "full"])


def test_right_player_wrong_day_is_not_relevant() -> None:
    """The failure mode that matters. A looser predicate would score this correct and
    the eval would report success while answering the wrong question."""
    wednesday = ROW.replace("Thursday", "Wednesday").replace("FULL", "LIMITED")
    assert not is_relevant(wednesday, ["Cameron Latu", "Thursday", "FULL"])


# --- metrics -----------------------------------------------------------------------


def test_reciprocal_rank_uses_first_relevant_position() -> None:
    result = evaluate_one("q", "q?", ["target"], ["no", "no", "target here", "target again"])
    assert result.first_position == 3
    assert result.reciprocal_rank == pytest.approx(1 / 3)
    assert result.total_relevant_found == 2


def test_a_miss_scores_zero_not_undefined() -> None:
    result = evaluate_one("q", "q?", ["target"], ["no", "nothing"])
    assert not result.hit
    assert result.reciprocal_rank == 0.0


def test_report_aggregates() -> None:
    report = Report(
        [
            evaluate_one("a", "?", ["x"], ["x", "b"]),  # rank 1
            evaluate_one("b", "?", ["x"], ["a", "b", "x"]),  # rank 3
            evaluate_one("c", "?", ["x"], ["a", "b"]),  # miss
        ]
    )
    assert report.n == 3
    assert report.hit_rate_at(1) == pytest.approx(1 / 3)
    assert report.hit_rate_at(3) == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert [m.question_id for m in report.misses] == ["c"]


def test_empty_report_does_not_divide_by_zero() -> None:
    report = Report([])
    assert report.mrr == 0.0
    assert report.hit_rate_at(5) == 0.0
