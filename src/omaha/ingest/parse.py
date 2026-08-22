"""Parsing fetched bytes into text plus structured tables.

Practice reports are tables, so the useful output is rows, not prose. But club sites
differ and reorganise, so every parser degrades to full text rather than failing —
a document with text and no tables is still worth storing and still chunkable.

`pdfplumber` rather than `docling`: docling is better on gnarly layout but pulls torch,
which has no x86_64 macOS wheels. Practice reports are plain tables and pdfplumber
handles them. Revisit if gamebooks prove too much.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import pdfplumber
import trafilatura
from selectolax.parser import HTMLParser

# Practice-report vocabulary. Participation is FULL / LIMITED / DNP; game status is
# Out / Doubtful / Questionable. Clubs abbreviate inconsistently.
PARTICIPATION = {
    "FULL": "FULL",
    "FP": "FULL",
    "FULL PARTICIPATION": "FULL",
    "LIMITED": "LIMITED",
    "LP": "LIMITED",
    "LIMITED PARTICIPATION": "LIMITED",
    "DNP": "DNP",
    "DID NOT PARTICIPATE": "DNP",
}
GAME_STATUS = {"OUT", "DOUBTFUL", "QUESTIONABLE", "QUES", "Q", "D", "O"}


@dataclass
class ParsedDocument:
    """What a parser produces. `tables` may be empty; `text` should never be."""

    text: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    parser: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def normalise_participation(value: str) -> str | None:
    """Map a cell to FULL / LIMITED / DNP, or None if it isn't a participation value."""
    key = re.sub(r"[^A-Z ]", "", value.strip().upper())
    return PARTICIPATION.get(key)


def looks_like_game_status(value: str) -> bool:
    return re.sub(r"[^A-Z]", "", value.strip().upper()) in GAME_STATUS


def parse_pdf(content: bytes) -> ParsedDocument:
    """Extract text and tables from a PDF.

    Tables come back as raw row lists with a page index. Interpreting which column is
    which is left to the caller — clubs order them differently and guessing silently is
    how you end up with a player's position in the injury column.
    """
    warnings: list[str] = []
    text_parts: list[str] = []
    tables: list[dict[str, Any]] = []

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page_no, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)

                for table_no, raw in enumerate(page.extract_tables()):
                    rows = [
                        [(cell or "").strip() for cell in row]
                        for row in raw
                        if any((cell or "").strip() for cell in row)
                    ]
                    if rows:
                        tables.append({"page": page_no, "index": table_no, "rows": rows})
    except Exception as exc:
        warnings.append(f"pdf parse failed: {exc!r}")

    return ParsedDocument(
        text="\n\n".join(p for p in text_parts if p).strip(),
        tables=tables,
        parser="pdfplumber",
        warnings=warnings,
    )


def parse_html(content: bytes, *, url: str | None = None) -> ParsedDocument:
    """Extract article text and any tables from HTML.

    trafilatura for prose (it strips nav and boilerplate well), selectolax for tables,
    since injury reports on club sites are usually real `<table>` markup.
    """
    warnings: list[str] = []
    html = content.decode("utf-8", errors="replace")

    text = trafilatura.extract(html, url=url, include_tables=True) or ""
    if not text.strip():
        # trafilatura declines on pages that are mostly table; fall back to raw text
        text = HTMLParser(html).text(separator="\n", strip=True)
        warnings.append("trafilatura returned nothing; used raw text")

    tables: list[dict[str, Any]] = []
    tree = HTMLParser(html)
    for table_no, table in enumerate(tree.css("table")):
        rows: list[list[str]] = []
        for tr in table.css("tr"):
            cells = [c.text(strip=True) for c in tr.css("th, td")]
            if any(cells):
                rows.append(cells)
        if rows:
            tables.append({"page": 0, "index": table_no, "rows": rows})

    return ParsedDocument(
        text=text.strip(), tables=tables, parser="trafilatura+selectolax", warnings=warnings
    )


def parse(content: bytes, *, content_type: str | None, url: str | None = None) -> ParsedDocument:
    """Dispatch on content type, sniffing the magic bytes when the header lies."""
    ct = (content_type or "").lower()
    if "pdf" in ct or content[:5] == b"%PDF-":
        return parse_pdf(content)
    return parse_html(content, url=url)
