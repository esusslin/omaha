"""Search endpoint behaviour that doesn't need a database.

Parameter validation and the graceful-degradation path are where this endpoint can
embarrass itself in a demo, so those are what's tested. The retrieval itself is covered
by `make eval` against a live corpus.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi import HTTPException

from omaha import search_api


class FakeHit:
    def __init__(self, chunk_id: int = 1, text: str = "row") -> None:
        self.chunk_id = chunk_id
        self.document_id = 7
        self.text = text
        self.score = 0.0328
        self.section_path = "injury_report > table 0 > row 3"
        self.doc_type = "injury_report"
        self.team = "PHI"
        self.source_url = "https://example.com/report"
        self.knowledge_time = dt.datetime(2025, 12, 18, 21, 15, tzinfo=dt.UTC)
        self.published_time = None
        self.dense_rank = 2
        self.lexical_rank = 1
        self.found_by = "both"


def test_serialise_shape() -> None:
    payload = search_api._serialise(FakeHit())
    assert payload["found_by"] == "both"
    assert payload["dense_rank"] == 2
    assert payload["knowledge_time"].startswith("2025-12-18")
    assert payload["published_time"] is None


@pytest.fixture(autouse=True)
def _clear_embedder_cache():
    search_api.get_embedder.cache_clear()
    yield
    search_api.get_embedder.cache_clear()


def _call(monkeypatch, *, embedder, mode="hybrid", as_of=None, captured=None):
    monkeypatch.setattr(search_api, "get_embedder", lambda: embedder)

    def fake(name):
        def inner(session, *args, **kwargs):
            if captured is not None:
                captured["fn"] = name
                captured["as_of"] = kwargs.get("as_of")
            return [FakeHit()]

        return inner

    monkeypatch.setattr(search_api, "lexical_search", fake("lexical"))
    monkeypatch.setattr(search_api, "dense_search", fake("dense"))
    monkeypatch.setattr(search_api, "hybrid_search", fake("hybrid"))

    return search_api.search(None, q="who is out", mode=mode, as_of=as_of)


def test_hybrid_falls_back_to_lexical_without_a_model(monkeypatch) -> None:
    """The demo path. Without this the endpoint 500s on any host lacking fastembed."""
    captured: dict = {}
    body = _call(monkeypatch, embedder=None, mode="hybrid", captured=captured)

    assert captured["fn"] == "lexical"
    assert body["mode_requested"] == "hybrid"
    assert body["mode_used"] == "lexical"
    assert body["dense_available"] is False


def test_response_admits_degradation_rather_than_hiding_it(monkeypatch) -> None:
    body = _call(monkeypatch, embedder=None, mode="dense")
    assert body["mode_used"] == "lexical"
    assert body["mode_requested"] == "dense"


def test_hybrid_used_when_model_present(monkeypatch) -> None:
    captured: dict = {}
    body = _call(monkeypatch, embedder=lambda t: [0.1] * 768, mode="hybrid", captured=captured)
    assert captured["fn"] == "hybrid"
    assert body["mode_used"] == "hybrid"
    assert body["dense_available"] is True


def test_lexical_requested_stays_lexical_even_with_a_model(monkeypatch) -> None:
    captured: dict = {}
    _call(monkeypatch, embedder=lambda t: [0.1] * 768, mode="lexical", captured=captured)
    assert captured["fn"] == "lexical"


# --- as_of parsing ------------------------------------------------------------------


def test_as_of_is_parsed_and_passed_through(monkeypatch) -> None:
    captured: dict = {}
    body = _call(monkeypatch, embedder=None, as_of="2025-12-19T17:00:00Z", captured=captured)

    assert captured["as_of"] == dt.datetime(2025, 12, 19, 17, 0, tzinfo=dt.UTC)
    assert body["as_of"].startswith("2025-12-19T17:00")


def test_naive_as_of_is_treated_as_utc(monkeypatch) -> None:
    """A naive timestamp silently compared against tz-aware rows is a leakage bug."""
    captured: dict = {}
    _call(monkeypatch, embedder=None, as_of="2025-12-19T17:00:00", captured=captured)
    assert captured["as_of"].tzinfo is not None


def test_bad_as_of_is_422_not_500(monkeypatch) -> None:
    with pytest.raises(HTTPException) as exc:
        _call(monkeypatch, embedder=None, as_of="last Tuesday")
    assert exc.value.status_code == 422


def test_absent_as_of_means_no_filter(monkeypatch) -> None:
    captured: dict = {}
    body = _call(monkeypatch, embedder=None, as_of=None, captured=captured)
    assert captured["as_of"] is None
    assert body["as_of"] is None


# --- the page -----------------------------------------------------------------------


def test_ui_is_self_contained() -> None:
    """No build step and no CDN — a portfolio page that needs npm install doesn't get
    looked at, and one that needs a third-party script breaks when that script moves."""
    page = search_api.ui()
    assert page.startswith("<!doctype html>")
    assert "<script" in page
    assert "http://" not in page.replace("http://www.w3.org", "")
    assert "cdn" not in page.lower()
