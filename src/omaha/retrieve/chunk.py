"""Chunking, structure-aware.

Most of what we ingest is tables, not prose, and the atomic unit of a practice report is
**one player's row**: "K. Allen, WR, hamstring, LIMITED, Wed". Splitting that document
into 400-token windows would smear three players across a chunk boundary and make
citation meaningless.

So: table rows become chunks; prose gets windowed. Every chunk keeps a `section_path`
identifying where it came from, and text chunks keep character offsets into
`parsed_text` so a claim can point at exact source text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omaha.ingest.parse import normalise_participation

# Prose windowing. Deliberately conservative — bge-base handles 512 tokens, and
# ~1,600 characters is comfortably inside that with room for the query.
WINDOW_CHARS = 1200
OVERLAP_CHARS = 200

# Doc types whose value is in their tables
TABULAR_KINDS = {"injury_report", "depth_chart", "inactives", "gamebook"}


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk before it hits the database."""

    ordinal: int
    text: str
    section_path: str
    span_start: int | None = None
    span_end: int | None = None


def _row_to_text(header: list[str] | None, row: list[str]) -> str:
    """Render a table row as a self-contained sentence.

    Header labels are folded in so the chunk is meaningful alone — "K. Allen WR
    Hamstring Limited" retrieves poorly, "Player: K. Allen | Pos: WR | Injury:
    Hamstring | Wed: Limited" retrieves well and reads well in a citation.
    """
    if header and len(header) == len(row):
        pairs = [f"{h.strip()}: {c.strip()}" for h, c in zip(header, row, strict=True) if c.strip()]
        return " | ".join(pairs)
    return " | ".join(c.strip() for c in row if c.strip())


def _looks_like_header(row: list[str]) -> bool:
    """A row with no participation values and short cells is probably a header."""
    if any(normalise_participation(c) for c in row):
        return False
    return all(len(c) <= 24 for c in row)


def chunk_tables(tables: list[dict[str, Any]], *, doc_type: str) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    ordinal = 0

    for table in tables:
        rows: list[list[str]] = table.get("rows", [])
        if not rows:
            continue

        header = rows[0] if _looks_like_header(rows[0]) else None
        body = rows[1:] if header else rows

        for row_no, row in enumerate(body):
            text = _row_to_text(header, row)
            if not text.strip():
                continue
            drafts.append(
                ChunkDraft(
                    ordinal=ordinal,
                    text=text,
                    section_path=f"{doc_type} > table {table.get('index', 0)} > row {row_no}",
                )
            )
            ordinal += 1

    return drafts


def chunk_text(text: str, *, doc_type: str, start_ordinal: int = 0) -> list[ChunkDraft]:
    """Window prose, breaking on paragraph boundaries where possible.

    Character offsets are real offsets into the text passed in, so a citation can point
    at the exact span.
    """
    drafts: list[ChunkDraft] = []
    ordinal = start_ordinal
    position = 0
    length = len(text)

    while position < length:
        end = min(position + WINDOW_CHARS, length)

        # prefer a paragraph break, then a sentence end, then a hard cut
        if end < length:
            for sep in ("\n\n", "\n", ". "):
                found = text.rfind(sep, position + WINDOW_CHARS // 2, end)
                if found != -1:
                    end = found + len(sep)
                    break

        window = text[position:end].strip()
        if window:
            drafts.append(
                ChunkDraft(
                    ordinal=ordinal,
                    text=window,
                    section_path=f"{doc_type} > text > chars {position}-{end}",
                    span_start=position,
                    span_end=end,
                )
            )
            ordinal += 1

        # Done once the window reaches the end. Without this, `end - OVERLAP_CHARS`
        # goes negative on short text, `position + 1` wins, and the loop emits one
        # chunk per character.
        if end >= length:
            break
        if end <= position:  # pathological; avoid an infinite loop
            break
        position = max(end - OVERLAP_CHARS, position + 1)

    return drafts


def chunk_document(
    *, doc_type: str, text: str, tables: list[dict[str, Any]] | None
) -> list[ChunkDraft]:
    """Chunk one document.

    Tabular documents chunk by row and skip the prose — the prose in a practice report
    is the legend, which adds noise to retrieval without adding information. Everything
    else windows the text.
    """
    tables = tables or []

    if doc_type in TABULAR_KINDS and tables:
        drafts = chunk_tables(tables, doc_type=doc_type)
        if drafts:
            return drafts

    return chunk_text(text, doc_type=doc_type)
