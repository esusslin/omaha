"""Day 2 tests. No network, no database — parsers and pure logic only.

The store-layer bitemporal behaviour gets integration tests once there's a test database
fixture; for now the invariant is documented in `store.py` and exercised by hand via
`omaha.ingest.run asof`.
"""

from __future__ import annotations

import datetime as dt

from omaha.ingest.fetch import FetchResult
from omaha.ingest.parse import (
    looks_like_game_status,
    normalise_participation,
    parse,
    parse_html,
)

NOW = dt.datetime(2026, 9, 10, 18, 0, tzinfo=dt.UTC)


# --- FetchResult ------------------------------------------------------------------


def test_content_hash_is_stable() -> None:
    a = FetchResult(url="u", status_code=200, fetched_at=NOW, content=b"hello")
    b = FetchResult(url="u", status_code=200, fetched_at=NOW, content=b"hello")
    assert a.content_hash == b.content_hash
    assert a.content_hash is not None
    assert len(a.content_hash) == 64


def test_content_hash_changes_with_content() -> None:
    a = FetchResult(url="u", status_code=200, fetched_at=NOW, content=b"wed")
    b = FetchResult(url="u", status_code=200, fetched_at=NOW, content=b"thu")
    assert a.content_hash != b.content_hash


def test_not_modified_is_not_an_error() -> None:
    r = FetchResult(url="u", status_code=304, fetched_at=NOW)
    assert r.not_modified
    assert not r.ok
    assert r.error is None


# --- participation vocabulary -----------------------------------------------------


def test_participation_normalisation() -> None:
    assert normalise_participation("DNP") == "DNP"
    assert normalise_participation("did not participate") == "DNP"
    assert normalise_participation(" Limited ") == "LIMITED"
    assert normalise_participation("FP") == "FULL"
    assert normalise_participation("Full Participation") == "FULL"


def test_participation_rejects_non_values() -> None:
    assert normalise_participation("hamstring") is None
    assert normalise_participation("WR") is None
    assert normalise_participation("") is None


def test_game_status_detection() -> None:
    assert looks_like_game_status("Questionable")
    assert looks_like_game_status("OUT")
    assert not looks_like_game_status("Achilles")


# --- HTML parsing -----------------------------------------------------------------

INJURY_TABLE = b"""
<html><body>
  <h1>Wednesday Injury Report</h1>
  <table>
    <tr><th>Player</th><th>Pos</th><th>Injury</th><th>Wed</th></tr>
    <tr><td>K. Allen</td><td>WR</td><td>Hamstring</td><td>Limited</td></tr>
    <tr><td>J. Smith</td><td>LT</td><td>Ankle</td><td>DNP</td></tr>
  </table>
</body></html>
"""


def test_html_tables_are_extracted() -> None:
    parsed = parse_html(INJURY_TABLE, url="https://example.com/injury-report")
    assert parsed.tables, "expected at least one table"
    rows = parsed.tables[0]["rows"]
    assert rows[0] == ["Player", "Pos", "Injury", "Wed"]
    assert ["K. Allen", "WR", "Hamstring", "Limited"] in rows


def test_html_participation_round_trip() -> None:
    """A row's status cell should normalise — this is the atom the chunker will use."""
    parsed = parse_html(INJURY_TABLE)
    rows = parsed.tables[0]["rows"]
    statuses = [normalise_participation(row[-1]) for row in rows[1:]]
    assert statuses == ["LIMITED", "DNP"]


def test_html_never_returns_empty_when_there_is_content() -> None:
    parsed = parse_html(INJURY_TABLE)
    assert not parsed.is_empty


# --- dispatch ---------------------------------------------------------------------


def test_parse_sniffs_pdf_magic_bytes_over_wrong_header() -> None:
    """Some club servers send PDFs as text/html. Trust the bytes."""
    parsed = parse(b"%PDF-1.4 garbage", content_type="text/html")
    assert parsed.parser == "pdfplumber"


def test_parse_dispatches_html_by_default() -> None:
    parsed = parse(INJURY_TABLE, content_type="text/html; charset=utf-8")
    assert parsed.parser == "trafilatura+selectolax"


def test_broken_pdf_warns_rather_than_raises() -> None:
    parsed = parse(b"%PDF-1.4 not really a pdf", content_type="application/pdf")
    assert parsed.warnings
    assert parsed.is_empty  # nothing extracted, but no exception


# --- text fingerprinting (semantic dedup) -----------------------------------------


def test_fingerprint_ignores_whitespace_reflow() -> None:
    from omaha.ingest.store import text_fingerprint

    a = text_fingerprint("K. Allen  WR  Hamstring  Limited")
    b = text_fingerprint("K. Allen\nWR\tHamstring\n\nLimited")
    assert a == b


def test_fingerprint_detects_a_real_change() -> None:
    from omaha.ingest.store import text_fingerprint

    wed = text_fingerprint("K. Allen WR Hamstring Limited")
    thu = text_fingerprint("K. Allen WR Hamstring DNP")
    assert wed != thu


def test_fingerprint_is_stable_across_calls() -> None:
    from omaha.ingest.store import text_fingerprint

    assert text_fingerprint("same text") == text_fingerprint("same text")


# --- cadence gating ---------------------------------------------------------------


def _source(**kw):
    from omaha.db.models import Source

    defaults = dict(
        name="s",
        kind="injury_report",
        url="https://example.com",
        enabled=True,
        cadence_seconds=3600,
        last_attempt_at=None,
    )
    defaults.update(kw)
    return Source(**defaults)


def test_never_attempted_is_due() -> None:
    from omaha.ingest.sweep import is_due

    assert is_due(_source(), NOW)


def test_recently_attempted_is_not_due() -> None:
    from omaha.ingest.sweep import is_due

    src = _source(last_attempt_at=NOW - dt.timedelta(minutes=10))
    assert not is_due(src, NOW)


def test_past_cadence_is_due() -> None:
    from omaha.ingest.sweep import is_due

    src = _source(last_attempt_at=NOW - dt.timedelta(hours=2))
    assert is_due(src, NOW)


def test_disabled_is_never_due() -> None:
    from omaha.ingest.sweep import is_due

    assert not is_due(_source(enabled=False), NOW)


def test_gating_uses_attempt_not_success() -> None:
    """A failing source must not be retried on every tick — that hammers a sick origin."""
    from omaha.ingest.sweep import is_due

    src = _source(last_attempt_at=NOW - dt.timedelta(minutes=5), last_success_at=None)
    assert not is_due(src, NOW)
