"""Index discovery and the club registry. No network.

The markup here mirrors the real Eagles injury-report index: a curated list of report
articles mixed in with features, video and photo galleries, and a cross-host link to the
opponent's site.
"""

from __future__ import annotations

import datetime as dt

from omaha.ingest.discover import (
    DiscoveredLink,
    discover_links,
    extract_published_time,
    parse_display_date,
)
from omaha.ingest.seeds import (
    CLUBS,
    UNAVAILABLE,
    all_seeds,
    injury_index_seeds,
    unavailable_source_names,
)

BASE = "https://www.philadelphiaeagles.com/team/injury-report/"

INDEX = """
<html><body>
<a href="/news/49ers-vs-eagles-injury-report-wild-card-round">49ers vs. Eagles Injury Report | Wild Card Round</a>
<a href="/news/eagles-at-bills-inactives-week-17">Eagles at Bills Inactives | Week 17</a>
<a href="/news/eagles-at-commanders-injury-report-week-16">Eagles at Commanders Injury Report</a>
<a href="/news/eagles-at-commanders-injury-report-week-16">Eagles at Commanders Injury Report</a>
<a href="/video/sirianni-press-conference-injury-report">Sirianni on the Injury Report</a>
<a href="/photos/photos-preparing-for-commanders">Photos: Preparing for the Commanders</a>
<a href="/news/spadaro-6-things-to-watch">Spadaro: 6 things to watch</a>
<a href="https://www.commanders.com/news/game-status-week-16">Game Status | Commanders</a>
<a href="#main-content">Skip to main content</a>
</body></html>
"""


def test_keeps_only_report_links() -> None:
    titles = [link.title for link in discover_links(INDEX, base_url=BASE)]
    assert "Eagles at Commanders Injury Report" in titles
    assert "Eagles at Bills Inactives | Week 17" in titles
    assert "Spadaro: 6 things to watch" not in titles


def test_excludes_video_and_photo_paths() -> None:
    """A video titled 'Injury Report' matches the title rule but has nothing to parse."""
    urls = [link.url for link in discover_links(INDEX, base_url=BASE)]
    assert not any("/video/" in u or "/photos/" in u for u in urls)


def test_deduplicates_repeated_links() -> None:
    urls = [link.url for link in discover_links(INDEX, base_url=BASE)]
    assert len(urls) == len(set(urls))


def test_relative_hrefs_are_resolved() -> None:
    assert all(link.url.startswith("https://") for link in discover_links(INDEX, base_url=BASE))


def test_cross_host_links_dropped_by_default() -> None:
    urls = [link.url for link in discover_links(INDEX, base_url=BASE)]
    assert not any("commanders.com" in u for u in urls)


def test_cross_host_links_kept_when_asked() -> None:
    urls = [link.url for link in discover_links(INDEX, base_url=BASE, same_host_only=False)]
    assert any("commanders.com" in u for u in urls)


def test_anchors_and_empty_hrefs_ignored() -> None:
    assert not any(
        link.url.endswith("#main-content") for link in discover_links(INDEX, base_url=BASE)
    )


def test_limit_is_respected() -> None:
    assert len(discover_links(INDEX, base_url=BASE, limit=2)) == 2


# --- two regressions from the first real run --------------------------------------

CONCATENATED = """
<a href="/news/commanders-vs-eagles-injury-report">newsCommanders vs. Eagles Injury ReportJan 02, 2026</a>
<a href="/news/falcons-injury-report-london">Falcons injury report: Drake London returns to practiceDec 17, 2025</a>
<a href="/news/week-18-injury-report-ravens">newsWeek 18 Injury Report (Ravens)Jan 02, 2026A look at player injuries</a>
"""


def test_title_matching_survives_concatenated_link_text() -> None:
    """Link text is assembled from sibling nodes, so the date runs straight on:
    "...Injury ReportJan 02, 2026". A trailing `\\b` in the title pattern finds a word
    character there and fails — which cut the Eagles index from 24 articles to 4 while
    still looking like it worked."""
    assert len(discover_links(CONCATENATED, base_url=BASE)) == 3


NAV = """
<a href="/team/injury-report/">Injury Report</a>
<a href="/team/transactions/">Transactions</a>
<a href="/team/depth-chart/">Depth Chart</a>
<a href="/news/eagles-at-bills-injury-report">Eagles at Bills Injury Report</a>
"""


def test_sub_nav_links_are_not_articles() -> None:
    """Every club page carries "Injury Report" in its sub-nav and footer, pointing back
    at the client-rendered widget. It matches the title rule perfectly and is pure
    navigation — following it re-fetches the page we're already on."""
    links = discover_links(NAV, base_url=BASE)
    assert len(links) == 1
    assert "/news/" in links[0].url


def test_slug_is_usable_as_a_name() -> None:
    link = DiscoveredLink(url="https://x.com/news/eagles-at-bills-inactives-week-17/", title="t")
    assert link.slug == "eagles-at-bills-inactives-week-17"


# --- publication time -------------------------------------------------------------


def test_published_time_from_meta() -> None:
    html = '<meta property="article:published_time" content="2025-12-18T21:15:00Z">'
    assert extract_published_time(html) == dt.datetime(2025, 12, 18, 21, 15, tzinfo=dt.UTC)


def test_published_time_from_json_ld() -> None:
    html = '<script type="application/ld+json">{"datePublished":"2025-12-18T21:15:00Z"}</script>'
    assert extract_published_time(html) == dt.datetime(2025, 12, 18, 21, 15, tzinfo=dt.UTC)


def test_published_time_absent_returns_none() -> None:
    """None beats a guess: this is the field a backtest trusts."""
    assert extract_published_time("<html><body>no dates here</body></html>") is None


def test_display_date_parses_club_format() -> None:
    parsed = parse_display_date("Eagles at Commanders Injury Report\nDec 18, 2025 at 04:15 PM")
    assert parsed is not None
    assert (parsed.year, parsed.month, parsed.day, parsed.hour) == (2025, 12, 18, 16)


def test_display_date_without_time() -> None:
    parsed = parse_display_date("Jan 10, 2026")
    assert parsed is not None and parsed.month == 1


def test_display_date_rejects_prose() -> None:
    assert parse_display_date("no date in this sentence") is None


# --- the registry -----------------------------------------------------------------


def test_all_32_clubs_present_and_distinct() -> None:
    assert len(CLUBS) == 32
    assert len({c.abbr for c in CLUBS}) == 32
    assert len({c.domain for c in CLUBS}) == 32


def test_seed_names_are_unique() -> None:
    names = [s.name for s in all_seeds()]
    assert len(names) == len(set(names))


def test_index_seeds_are_index_kind() -> None:
    """Kind drives sweep routing — if these register as `injury_report` they get
    treated as documents and quietly collect the page legend."""
    seeds = injury_index_seeds()
    assert len(seeds) == 32 - sum(1 for _, kind in UNAVAILABLE if kind == "injury_index")
    assert {s.kind for s in seeds} == {"injury_index"}


def test_index_cadence_is_not_hourly() -> None:
    assert all(s.cadence_seconds >= 3600 for s in all_seeds())


def test_unavailable_clubs_are_not_seeded() -> None:
    """Two clubs don't serve what the shared CMS implies they should: Dallas has no
    transactions page at all, and Detroit returns 404 for our user-agent on the injury
    index. Seeding them anyway means two sources whose `consecutive_failures` climb
    forever and two red rows on `/health` that nobody intends to fix — which trains you
    to ignore the endpoint."""
    names = {s.name for s in all_seeds()}
    assert "dal-transactions" not in names
    assert "det-injury-index" not in names

    # ...but the clubs are still covered for everything that does work.
    assert "dal-injury-index" in names
    assert "det-transactions" in names


def test_every_exception_carries_a_reason() -> None:
    """An exceptions table without reasons is folklore. Six months from now the question
    is 'why is Dallas missing?', and the answer has to live next to the exclusion."""
    assert all(reason.strip() for reason in UNAVAILABLE.values())
    assert all(len(reason) > 30 for reason in UNAVAILABLE.values())


def test_unavailable_names_match_real_seed_names() -> None:
    """`unavailable_source_names()` drives the disable step in `run seed`, so a typo
    there is silent: the stale source stays enabled and keeps failing. This pins the
    naming convention to the one the seeds actually use."""
    from omaha.ingest.seeds import CLUBS as _CLUBS

    all_possible = {f"{club.abbr.lower()}-injury-index" for club in _CLUBS} | {
        f"{club.abbr.lower()}-transactions" for club in _CLUBS
    }
    assert set(unavailable_source_names()) <= all_possible
