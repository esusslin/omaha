"""Following an index page to the documents it links.

Every source until now was one URL holding one thing that changes over time — poll it,
diff it, store a snapshot when it moves. Injury reports don't work that way. The club
publishes a *new article per week*, and the injury-report page is an index listing them.

So this adds a second shape: an **index source**. We poll the index, extract the article
links whose titles look like reports, and fetch each one we haven't seen. Articles are
immutable once published, so the interesting question is "is this URL new to us?" rather
than "has this URL changed?"

Two details that matter:

**Titles, not URL patterns.** Club slugs are editorial free-text
(`eagles-at-commanders-injury-report-nfl-week-16-2025-lane-johnson-jayden-daniels`), so
matching on the URL is guesswork. The link text is consistently "<Matchup> Injury Report"
or "<Matchup> Inactives", which is what we match on.

**published_time finally earns its column.** These articles carry a real publication
timestamp. It goes in `published_time`; `knowledge_time` stays the moment *we* fetched
it. Backfilled history is therefore honestly marked as something we learned late, which
is what keeps a backtest from quietly reading the future.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from selectolax.parser import HTMLParser

# Link titles worth following. Deliberately narrow: a club news index is mostly
# features, video and photo galleries, and fetching all of it is both rude and useless.
#
# No trailing `\b`. Link text is assembled from sibling elements and comes out
# concatenated — "Commanders vs. Eagles Injury ReportJan 02, 2026" — so a word boundary
# after "report" finds a word character and fails. That one anchor silently cut the
# Eagles index from 24 articles to 4, keeping only the titles that happened to be
# followed by punctuation.
REPORT_TITLE = re.compile(
    r"\b(injury\s+report|inactives|practice\s+report|game\s+status|roster\s+moves?)",
    re.IGNORECASE,
)

# Things that look like reports but aren't documents we can parse.
#
# `/team/` matters more than it looks: every club page carries "Injury Report",
# "Transactions" and "Depth Chart" in its sub-nav and footer, pointing back at the
# client-rendered widget pages. Those match the title rule perfectly and are pure
# navigation — they're why almost every club reported "1 linked" on the first run.
EXCLUDE_PATH = re.compile(r"/(video|photos|audio|podcast|team)/", re.IGNORECASE)

_META_PUBLISHED = (
    "article:published_time",
    "og:published_time",
    "datePublished",
    "pubdate",
)

_DATE_TEXT = re.compile(
    r"\b([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{4})(?:\s+at\s+(\d{1,2}):(\d{2})\s*([AP]M))?"
)

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}


@dataclass(frozen=True)
class DiscoveredLink:
    """An article the index points at."""

    url: str
    title: str

    @property
    def slug(self) -> str:
        """Stable short name, for a Source or a log line."""
        return urlparse(self.url).path.rstrip("/").rsplit("/", 1)[-1][:96]


def discover_links(
    html: str,
    *,
    base_url: str,
    same_host_only: bool = True,
    limit: int = 200,
) -> list[DiscoveredLink]:
    """Extract report links from an index page.

    Deduplicated by URL, preserving document order — club indexes list newest first, so
    the caller gets the most recent work first and can stop early on a partial backfill.
    """
    tree = HTMLParser(html)
    base_host = urlparse(base_url).netloc

    seen: set[str] = set()
    found: list[DiscoveredLink] = []

    for node in tree.css("a[href]"):
        href = (node.attributes.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue

        title = node.text(strip=True) or (node.attributes.get("title") or "").strip()
        if not REPORT_TITLE.search(title):
            continue

        url = urljoin(base_url, href).split("#", 1)[0]
        if EXCLUDE_PATH.search(urlparse(url).path):
            continue
        if same_host_only and urlparse(url).netloc != base_host:
            continue
        if url in seen:
            continue

        seen.add(url)
        found.append(DiscoveredLink(url=url, title=title))
        if len(found) >= limit:
            break

    return found


def extract_published_time(html: str) -> dt.datetime | None:
    """Best effort at the article's own publication timestamp.

    Returns None rather than guessing. A wrong `published_time` is worse than a missing
    one — it is the field a backtest trusts to decide what was knowable.
    """
    tree = HTMLParser(html)

    for node in tree.css("meta"):
        attrs = node.attributes
        key = attrs.get("property") or attrs.get("name") or attrs.get("itemprop")
        content = attrs.get("content")
        if key in _META_PUBLISHED and content:
            parsed = _parse_iso(content)
            if parsed:
                return parsed

    for node in tree.css('script[type="application/ld+json"]'):
        parsed = _from_json_ld(node.text())
        if parsed:
            return parsed

    for node in tree.css("time[datetime]"):
        parsed = _parse_iso(node.attributes.get("datetime") or "")
        if parsed:
            return parsed

    return None


def parse_display_date(text: str) -> dt.datetime | None:
    """Parse the human date clubs print above the article: 'Dec 18, 2025 at 04:15 PM'.

    Naive — the club prints local time without a zone — so callers should treat this as a
    fallback for `published_time` only, never for `knowledge_time`.
    """
    match = _DATE_TEXT.search(text)
    if not match:
        return None

    month_name, day, year, hour, minute, meridiem = match.groups()
    month = _MONTHS.get(month_name)
    if month is None:
        return None

    if hour is None:
        return dt.datetime(int(year), month, int(day), tzinfo=dt.UTC)

    hour_i = int(hour) % 12
    if meridiem.upper() == "PM":
        hour_i += 12
    return dt.datetime(int(year), month, int(day), hour_i, int(minute), tzinfo=dt.UTC)


def _parse_iso(value: str) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def _from_json_ld(blob: str) -> dt.datetime | None:
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return None

    candidates = data if isinstance(data, list) else [data]
    for item in candidates:
        if not isinstance(item, dict):
            continue
        for key in ("datePublished", "dateCreated", "uploadDate"):
            if isinstance(item.get(key), str):
                parsed = _parse_iso(item[key])
                if parsed:
                    return parsed
    return None
