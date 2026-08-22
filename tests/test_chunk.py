"""Day 4 — chunking. Pure functions, no database, no model download."""

from __future__ import annotations

import itertools

from omaha.retrieve.chunk import (
    OVERLAP_CHARS,
    WINDOW_CHARS,
    chunk_document,
    chunk_tables,
    chunk_text,
)

INJURY_TABLE = [
    {
        "index": 0,
        "rows": [
            ["Player", "Pos", "Injury", "Wed"],
            ["K. Allen", "WR", "Hamstring", "Limited"],
            ["J. Smith", "LT", "Ankle", "DNP"],
        ],
    }
]


def test_one_chunk_per_player_row() -> None:
    """The atom of a practice report is a player, not a paragraph."""
    drafts = chunk_tables(INJURY_TABLE, doc_type="injury_report")
    assert len(drafts) == 2


def test_header_labels_are_folded_into_the_chunk() -> None:
    """A chunk has to make sense alone — it's what gets retrieved and cited."""
    drafts = chunk_tables(INJURY_TABLE, doc_type="injury_report")
    assert "Player: K. Allen" in drafts[0].text
    assert "Wed: Limited" in drafts[0].text


def test_header_row_is_not_a_chunk() -> None:
    drafts = chunk_tables(INJURY_TABLE, doc_type="injury_report")
    assert not any("Pos: Pos" in d.text for d in drafts)


def test_section_path_locates_the_row() -> None:
    drafts = chunk_tables(INJURY_TABLE, doc_type="injury_report")
    assert drafts[0].section_path == "injury_report > table 0 > row 0"
    assert drafts[1].section_path == "injury_report > table 0 > row 1"


def test_ordinals_are_contiguous() -> None:
    drafts = chunk_tables(INJURY_TABLE, doc_type="injury_report")
    assert [d.ordinal for d in drafts] == [0, 1]


def test_a_row_without_a_header_still_chunks() -> None:
    tables = [{"index": 0, "rows": [["K. Allen", "WR", "Hamstring", "DNP"]]}]
    drafts = chunk_tables(tables, doc_type="injury_report")
    assert len(drafts) == 1
    assert "K. Allen" in drafts[0].text


# --- prose windowing --------------------------------------------------------------


def test_short_text_is_one_chunk() -> None:
    drafts = chunk_text("A coach said something brief.", doc_type="transcript")
    assert len(drafts) == 1
    assert drafts[0].span_start == 0


def test_long_text_windows_with_overlap() -> None:
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20))
    drafts = chunk_text(text, doc_type="transcript")
    assert len(drafts) > 1
    # windows advance but overlap
    for a, b in itertools.pairwise(drafts):
        assert b.span_start is not None and a.span_end is not None
        assert b.span_start < a.span_end
        assert b.span_start > a.span_start


def test_spans_point_at_real_text() -> None:
    """Offsets must be usable for citation — slice the source and get the chunk back."""
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    drafts = chunk_text(text, doc_type="transcript")
    for d in drafts:
        assert d.span_start is not None and d.span_end is not None
        assert text[d.span_start : d.span_end].strip() == d.text


def test_windows_respect_the_size_bound() -> None:
    text = "word " * 5000
    drafts = chunk_text(text, doc_type="transcript")
    assert all(len(d.text) <= WINDOW_CHARS + OVERLAP_CHARS for d in drafts)


# --- dispatch ---------------------------------------------------------------------


def test_tabular_documents_chunk_by_row_not_prose() -> None:
    """A practice report's prose is the legend — noise in retrieval."""
    drafts = chunk_document(
        doc_type="injury_report",
        text="INJURY REPORT Legend DNP - Did not participate " * 20,
        tables=INJURY_TABLE,
    )
    assert len(drafts) == 2
    assert all("Legend" not in d.text for d in drafts)


def test_tabular_document_without_tables_falls_back_to_text() -> None:
    drafts = chunk_document(doc_type="injury_report", text="No table today.", tables=[])
    assert len(drafts) == 1
    assert "No table" in drafts[0].text


def test_prose_document_ignores_tables_flag() -> None:
    drafts = chunk_document(doc_type="transcript", text="Coach said a thing.", tables=None)
    assert len(drafts) == 1
