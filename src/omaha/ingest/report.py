"""Extracting structured injury rows from prose-format club reports.

**Why this exists.** The obvious source — `/team/injury-report/` on a club site — is
rendered client-side. Fetching it returns the week selector and the legend and no player
rows, in August and in November alike. Verified against two clubs; the table is never in
the HTML. A collector pointed at that URL looks healthy forever and collects nothing.

What *is* server-rendered is the club's weekly injury-report news article, and it turns
out to be richer than the widget: one article carries both teams and the full
Tuesday/Wednesday/Thursday progression, which is exactly the day-over-day trajectory the
bitemporal store exists to preserve.

The catch is that those articles contain no `<table>` markup. The data is lines of prose
under headings::

    Thursday's Injury Report
    Eagles Injury Report
    Out
    DT Jalen Carter (Shoulders/Did Not Participate)
    T Lane Johnson (Foot/Did Not Participate)

So this module reconstructs rows from text. It deliberately emits the same
``{"index": .., "rows": [...]}`` shape that `parse_html` produces for real tables, so
the row-per-player chunker downstream needs no changes at all.

**On robustness.** Headings are matched on their text, not on any markup or markdown
around them, because whether `###` and `**` survive extraction depends on trafilatura's
output mode. Player lines are matched on a strict shape instead, which is the part that
is genuinely stable across clubs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from omaha.ingest.parse import normalise_participation

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# "Thursday's Injury Report" / "Wednesday Injury Report"
_DAY_HEADING = re.compile(
    # U+2019 is the typographic apostrophe. Club CMSes emit both it and plain ASCII
    # U+0027, sometimes in the same document, so the class has to accept both. Written
    # as an escape rather than the character itself: the two are indistinguishable on
    # screen, which is a nasty thing to leave in a regex someone edits later.
    rf"^\W*({'|'.join(WEEKDAYS)})['\u2019]?s?\s+(?:Injury|Practice)\s+Report\W*$",
    re.IGNORECASE,
)

# Section headings naming a team. Rather than guess at the shape, match against the 32
# real nicknames we already maintain in `seeds`.
#
# Regex alone kept getting this wrong. `^[A-Z]` silently dropped every 49ers heading —
# "49ers" starts with a digit — and anchoring on "<Team> Injury Report$" missed both
# "Injury Report Ahead of Week 12" and Pittsburgh's "Week 17 Injury Report (Browns)",
# where the team is in trailing parentheses. A closed vocabulary handles all three and
# cannot invent a team that doesn't exist.
# "Game Status Report" is the Friday/Saturday heading and carries the designations that
# decide whether a player suits up — the most valuable rows in the document. Omitting it
# here is what left 10.9% of rows unattributed.
_REPORT_HEADING = re.compile(r"(?:Injury|Practice|Game\s+Status)\s+Report", re.IGNORECASE)

_TEAM_NICKNAMES: tuple[str, ...] = ()


def _nicknames() -> tuple[str, ...]:
    """Club nicknames, longest first so 'Football Team' beats 'Team' on a prefix match."""
    global _TEAM_NICKNAMES
    if not _TEAM_NICKNAMES:
        from omaha.ingest.seeds import CLUBS

        _TEAM_NICKNAMES = tuple(sorted((c.name for c in CLUBS), key=len, reverse=True))
    return _TEAM_NICKNAMES


def parse_team_heading(line: str) -> str | None:
    """The team a section heading is about, or None if it isn't one.

    Returns None when a line names *two* clubs: that's the article title
    ("Eagles at Commanders Injury Report"), not a section heading, and letting it set
    the team would attribute the first block of rows to whichever side was named first.
    """
    if len(line) > 120 or not _REPORT_HEADING.search(line):
        return None

    found = [
        nickname
        for nickname in _nicknames()
        if re.search(rf"\b{re.escape(nickname)}\b", line, re.IGNORECASE)
    ]
    if len(found) != 1:
        return None
    return found[0]


# A status group label on its own line: "Out", "Questionable", "Did Not Participate"
_GROUP_LABEL = re.compile(
    r"^\W*(Out|Doubtful|Questionable|Did Not Participate|Limited Participation|"
    r"Full Participation|DNP|LP|FP)\W*$",
    re.IGNORECASE,
)

# "DT Jalen Carter (Shoulders/Did Not Participate)"
#
# Position allows a slash — "G/T Trey Pipkins III" is a real listing, and requiring
# 1-3 plain capitals dropped every dual-position player without a trace.
#
# The parenthetical is optional. Under a game-status group a club may write just
# "TE Lucas Krull", because the group heading already says Out. Requiring parentheses
# discarded exactly the rows where the status was most certain.
_POSITION = r"[A-Z]{1,3}(?:/[A-Z]{1,3})?"
_NAME = r"[A-Z0-9][\w'\u2019.-]*(?:\s+[A-Z0-9][\w'\u2019.-]*){0,3}"

_PLAYER_LINE = re.compile(
    rf"^\W*(?P<pos>{_POSITION})\s+(?P<player>{_NAME})\s*\((?P<detail>[^)]*)\)\W*$"
)

# Same shape with no detail. Kept separate and length-bounded so it can't start
# swallowing prose — a sentence beginning "DT Jalen Carter was limited..." must not
# read as a row.
_PLAYER_LINE_BARE = re.compile(rf"^\W*(?P<pos>{_POSITION})\s+(?P<player>{_NAME})\W*$")

GAME_STATUSES = {"OUT", "DOUBTFUL", "QUESTIONABLE"}

COLUMNS = ["Team", "Day", "Pos", "Player", "Injury", "Practice", "Status"]


@dataclass(frozen=True)
class InjuryRow:
    """One player's entry on one day's report."""

    team: str | None
    day: str | None
    position: str
    player: str
    injury: str | None
    participation: str | None
    """FULL / LIMITED / DNP, when the report says."""
    game_status: str | None
    """OUT / DOUBTFUL / QUESTIONABLE, when the report says."""

    def as_cells(self) -> list[str]:
        return [
            self.team or "",
            self.day or "",
            self.position,
            self.player,
            self.injury or "",
            self.participation or "",
            self.game_status or "",
        ]


def _split_detail(detail: str) -> tuple[str | None, str | None]:
    """Split the parenthesised part into (injury, participation).

    Clubs write `(Foot/Did Not Participate)` on game-status groups and plain `(Foot)`
    under participation groups. Sometimes the second part is not a participation value
    at all — `(Calf/Rest)` means a calf injury and a scheduled rest day — so this tests
    each part rather than assuming the last one is a status.
    """
    parts = [p.strip() for p in detail.split("/") if p.strip()]
    if not parts:
        return None, None

    participation = None
    injury_parts: list[str] = []
    for part in parts:
        mapped = normalise_participation(part)
        if mapped and participation is None:
            participation = mapped
        else:
            injury_parts.append(part)

    injury = "/".join(injury_parts) if injury_parts else None
    return injury, participation


def extract_injury_rows(text: str) -> list[InjuryRow]:
    """Pull player rows out of a club injury-report article.

    Day and team are carried forward from the most recent heading, because the article
    states each once and then lists under it.
    """
    rows: list[InjuryRow] = []
    day: str | None = None
    team: str | None = None
    group_participation: str | None = None
    group_status: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        day_match = _DAY_HEADING.match(line)
        if day_match:
            day = day_match.group(1).capitalize()
            # a new day resets the group, but not the team — the article repeats the
            # team heading under each day anyway
            group_participation = group_status = None
            continue

        group_match = _GROUP_LABEL.match(line)
        if group_match:
            label = group_match.group(1)
            if label.upper() in GAME_STATUSES:
                group_status = label.upper()
                group_participation = None
            else:
                group_participation = normalise_participation(label)
                group_status = None
            continue

        player_match = _PLAYER_LINE.match(line)
        bare_match = None
        if player_match is None and len(line) <= 48 and (group_status or group_participation):
            # Only accept a detail-less row inside a status group: the group heading is
            # what makes it meaningful, and the constraint keeps prose out.
            bare_match = _PLAYER_LINE_BARE.match(line)

        match = player_match or bare_match
        if match:
            injury, participation = (
                _split_detail(match.group("detail")) if player_match else (None, None)
            )
            rows.append(
                InjuryRow(
                    team=team,
                    day=day,
                    position=match.group("pos"),
                    player=match.group("player"),
                    injury=injury,
                    participation=participation or group_participation,
                    game_status=group_status,
                )
            )
            continue

        # Checked last: heading detection is loose enough to swallow player lines and
        # group labels if it gets first refusal.
        heading_team = parse_team_heading(line)
        if heading_team:
            team = heading_team
            group_participation = group_status = None

    return rows


def extract_injury_tables(text: str) -> list[dict[str, object]]:
    """Rows in the same shape `parse_html` emits for real `<table>` markup.

    Returning the parser's native table shape is the whole trick: the row-per-player
    chunker consumes this untouched, so recovering structure from prose costs nothing
    downstream.
    """
    rows = extract_injury_rows(text)
    if not rows:
        return []
    return [
        {
            "page": 0,
            "index": 0,
            "rows": [list(COLUMNS)] + [row.as_cells() for row in rows],
            "derived": "prose_injury_report",
        }
    ]


def looks_like_injury_report(text: str) -> bool:
    """Cheap gate so we don't run the extractor over every page we fetch."""
    head = text[:4000]
    return bool(re.search(r"(Injury|Practice)\s+Report", head, re.IGNORECASE))
