"""The records endpoint, and specifically its one non-obvious guarantee.

Most of this surface is a database query and doesn't need defending. What does need
defending is `knowledge`: the field that tells a consumer whether an empty result means
"nobody is injured" or "we have no idea". Those are the same JSON otherwise, and
conflating them is the mistake that produced a 41% spurious downgrade rate in
`the-algo`'s red team when weather data was absent.

No database required — `_knowledge_state` takes a session but the branch logic is driven
entirely by source freshness, which is faked here with plain objects.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest

from omaha import records_api


@dataclass
class FakeSource:
    """Just the three fields `_knowledge_state` reads."""

    last_success_at: dt.datetime | None
    cadence_seconds: int = 3600


class FakeSession:
    def __init__(self, sources: list[FakeSource]) -> None:
        self._sources = sources

    def scalars(self, _statement: object) -> FakeSession:
        return self

    def all(self) -> list[FakeSource]:
        return self._sources


def state(sources: list[FakeSource], as_of: dt.datetime | None = None) -> dict:
    return records_api._knowledge_state(FakeSession(sources), "PHI", as_of)  # type: ignore[arg-type]


def ago(seconds: float) -> dt.datetime:
    return dt.datetime.now(dt.UTC) - dt.timedelta(seconds=seconds)


# --- the distinction that matters ---------------------------------------------------


def test_fresh_sources_mean_an_empty_list_is_information() -> None:
    """All sources fresh and no records: the player is not on the report. That is a
    fact about the player, and a consumer is entitled to act on it."""
    result = state([FakeSource(ago(60)), FakeSource(ago(120))])
    assert result["knowledge"] == "complete"
    assert result["fresh"] == 2


def test_a_dead_scraper_is_not_a_clean_bill_of_health() -> None:
    """Every source overdue: an empty list says nothing whatsoever. This is the case the
    endpoint exists for — without it, a broken collector looks exactly like a healthy
    roster."""
    result = state([FakeSource(ago(99_999)), FakeSource(ago(99_999))])
    assert result["knowledge"] == "unknown"


def test_partial_coverage_is_reported_as_partial() -> None:
    """Some fresh, some stale. Records may be missing, so absence must be treated as
    unknown rather than as health — but what *is* returned is still usable."""
    result = state([FakeSource(ago(60)), FakeSource(ago(99_999))])
    assert result["knowledge"] == "partial"
    assert result["stale"] == 1


def test_a_source_that_never_succeeded_counts_against_confidence() -> None:
    result = state([FakeSource(ago(60)), FakeSource(None)])
    assert result["knowledge"] == "partial"
    assert result["never_succeeded"] == 1


def test_no_sources_at_all_is_unknown_not_complete() -> None:
    """An unregistered team must not report 'complete' — the most dangerous possible
    answer, since it asserts health about a club we have never once fetched."""
    assert state([])["knowledge"] == "unknown"


def test_staleness_tolerates_one_missed_poll_but_not_two() -> None:
    """A single missed poll is jitter; two is a pattern. The multiple is the whole
    definition of stale here, so it's worth pinning."""
    cadence = 3600
    assert state([FakeSource(ago(cadence * 1.5), cadence)])["knowledge"] == "complete"
    assert state([FakeSource(ago(cadence * 2.5), cadence)])["knowledge"] == "unknown"


# --- point-in-time ------------------------------------------------------------------


def test_a_historical_query_does_not_claim_present_freshness() -> None:
    """Sources being fresh *now* says nothing about whether collection was working last
    December. Claiming 'complete' for a backtest window would be a guarantee we cannot
    make, so the endpoint names the situation and hands the judgement back."""
    result = state([FakeSource(ago(60))], as_of=dt.datetime(2025, 12, 19, tzinfo=dt.UTC))
    assert result["knowledge"] == "as_of_historical"
    assert "past instant" in result["reason"]


# --- input handling -----------------------------------------------------------------


def test_a_malformed_timestamp_is_rejected_rather_than_ignored() -> None:
    """Silently dropping an unparseable `as_of` would return unfiltered data to a caller
    who asked for a point-in-time view — leakage delivered through a typo."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        records_api._parse_as_of("last tuesday")
    assert exc.value.status_code == 422


def test_naive_timestamps_are_treated_as_utc() -> None:
    parsed = records_api._parse_as_of("2025-12-19T17:00:00")
    assert parsed is not None and parsed.tzinfo is not None


def test_trailing_z_is_accepted() -> None:
    parsed = records_api._parse_as_of("2025-12-19T17:00:00Z")
    assert parsed is not None and parsed.utcoffset() == dt.timedelta(0)


# --- rate limiting ------------------------------------------------------------------


class FakeRequest:
    def __init__(self, ip: str) -> None:
        self.headers = {"x-forwarded-for": ip}
        self.client = None


def test_the_demo_endpoints_are_rate_limited() -> None:
    """The demo URL is in a public README with no auth. Without a limit, one loop makes
    it unavailable to everyone else and runs up a hosting bill."""
    from fastapi import HTTPException

    from omaha.search_api import RATE_LIMIT_REQUESTS, _hits, rate_limit

    _hits.clear()
    request = FakeRequest("203.0.113.7")
    for _ in range(RATE_LIMIT_REQUESTS):
        rate_limit(request)  # type: ignore[arg-type]

    with pytest.raises(HTTPException) as exc:
        rate_limit(request)  # type: ignore[arg-type]
    assert exc.value.status_code == 429
    assert "Retry-After" in (exc.value.headers or {})


def test_callers_are_counted_separately() -> None:
    """Behind Railway's proxy every visitor shares one socket address, so the limiter
    keys on X-Forwarded-For. Getting this wrong would rate-limit the whole internet as a
    single caller the moment one person ran a loop."""
    from omaha.search_api import RATE_LIMIT_REQUESTS, _hits, rate_limit

    _hits.clear()
    for _ in range(RATE_LIMIT_REQUESTS):
        rate_limit(FakeRequest("198.51.100.1"))  # type: ignore[arg-type]
    # A different caller is unaffected — no exception.
    rate_limit(FakeRequest("198.51.100.2"))  # type: ignore[arg-type]
