"""Tests for prose injury-report extraction, against a real club article.

The fixture is the structured listing portion of the Eagles' Week 16 2025 report — a
document that actually exists, with the mess that implies: a typo ("Illnesss"), rest days
recorded as injuries, a suffix in a player name, an apostrophe in another, lowercase
injury names, and two teams interleaved across three days.

Everything before this was tested against HTML I wrote myself, which only ever proves the
parser agrees with my imagination.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omaha.ingest.report import (
    extract_injury_rows,
    extract_injury_tables,
    looks_like_injury_report,
    parse_team_heading,
)

FIXTURE = Path(__file__).parent / "fixtures" / "eagles_commanders_2025_week16.txt"


@pytest.fixture(scope="module")
def rows():
    return extract_injury_rows(FIXTURE.read_text())


def _find(rows, player, day):
    matches = [r for r in rows if r.player == player and r.day == day]
    assert matches, f"no row for {player} on {day}"
    return matches[0]


# --- the basic claim --------------------------------------------------------------


def test_extracts_every_player_row(rows) -> None:
    """50 entries: 3 days x 2 teams, counted by hand off the published article."""
    assert len(rows) == 50


def test_no_row_is_missing_a_player_or_position(rows) -> None:
    assert all(r.player and r.position for r in rows)


def test_gate_recognises_the_document(_=None) -> None:
    assert looks_like_injury_report(FIXTURE.read_text())
    assert not looks_like_injury_report("Weekly practice notes and photo gallery")


# --- day attribution: the reason this document is worth having ---------------------


def test_all_three_days_are_captured(rows) -> None:
    assert {r.day for r in rows} == {"Tuesday", "Wednesday", "Thursday"}


def test_a_player_progression_across_days(rows) -> None:
    """Latu goes DNP -> limited -> full-and-questionable. This trajectory is the point."""
    assert _find(rows, "Cameron Latu", "Tuesday").participation == "DNP"
    assert _find(rows, "Cameron Latu", "Wednesday").participation == "LIMITED"

    thursday = _find(rows, "Cameron Latu", "Thursday")
    assert thursday.participation == "FULL"
    assert thursday.game_status == "QUESTIONABLE"


# --- team attribution -------------------------------------------------------------


def test_both_teams_are_attributed(rows) -> None:
    assert {r.team for r in rows} == {"Eagles", "Commanders"}


def test_team_does_not_bleed_across_headings(rows) -> None:
    """Daniels is a Commander on every day. A carried-forward team is easy to get wrong."""
    daniels = [r for r in rows if r.player == "Jayden Daniels"]
    assert len(daniels) == 3
    assert {r.team for r in daniels} == {"Commanders"}


# --- the parenthesised detail -----------------------------------------------------


def test_game_status_group_splits_injury_from_participation(rows) -> None:
    carter = _find(rows, "Jalen Carter", "Thursday")
    assert carter.game_status == "OUT"
    assert carter.participation == "DNP"
    assert carter.injury == "Shoulders"


def test_multiword_injury_survives(rows) -> None:
    daniels = _find(rows, "Jayden Daniels", "Thursday")
    assert daniels.injury == "Left Elbow"
    assert daniels.participation == "LIMITED"


def test_rest_is_not_mistaken_for_a_participation_value(rows) -> None:
    """`(Calf/Rest)` is a calf injury plus a rest day, not a status. Assuming the last
    slash-part is always a status would silently drop the calf."""
    dickerson = _find(rows, "Landon Dickerson", "Wednesday")
    assert dickerson.injury == "Calf/Rest"
    assert dickerson.participation == "DNP"  # inherited from the group heading


def test_participation_inherited_from_group_when_line_omits_it(rows) -> None:
    barkley = _find(rows, "Saquon Barkley", "Wednesday")
    assert barkley.participation == "FULL"
    assert barkley.game_status is None


def test_lowercase_injury_is_preserved_not_normalised_away(rows) -> None:
    assert _find(rows, "Tank Bigsby", "Wednesday").injury == "illness"


# --- names that break naive regexes -----------------------------------------------


def test_name_with_apostrophe(rows) -> None:
    newton = _find(rows, "Jer'Zhan Newton", "Thursday")
    assert newton.position == "DT"
    assert newton.injury == "Illnesss"  # sic — the club's typo, preserved


def test_name_with_suffix(rows) -> None:
    assert _find(rows, "Chris Rodriguez Jr.", "Wednesday").position == "RB"


def test_prose_footnote_is_not_parsed_as_a_player(rows) -> None:
    assert not any("walkthrough" in r.player.lower() for r in rows)


# --- the shape handoff ------------------------------------------------------------


def test_tables_match_the_shape_the_chunker_expects() -> None:
    """Emitting the parser's native table shape is what lets the row chunker work
    unchanged. If this drifts, chunking silently falls back to prose windowing."""
    tables = extract_injury_tables(FIXTURE.read_text())
    assert len(tables) == 1

    table = tables[0]
    assert set(table) >= {"page", "index", "rows"}

    header, *body = table["rows"]
    assert header == ["Team", "Day", "Pos", "Player", "Injury", "Practice", "Status"]
    assert len(body) == 50
    assert all(len(row) == len(header) for row in body)


def test_rows_chunk_one_per_player() -> None:
    """End to end: the real document produces one retrievable chunk per player-day."""
    from omaha.retrieve.chunk import chunk_document

    tables = extract_injury_tables(FIXTURE.read_text())
    drafts = chunk_document(doc_type="injury_report", text="ignored", tables=tables)

    assert len(drafts) == 50
    carter = [d for d in drafts if "Jalen Carter" in d.text and "Thursday" in d.text]
    assert len(carter) == 1
    # header labels folded in, so the chunk reads as a standalone fact
    assert "Player: Jalen Carter" in carter[0].text
    assert "Status: OUT" in carter[0].text


def test_empty_text_yields_no_tables() -> None:
    assert extract_injury_tables("") == []
    assert extract_injury_tables("Just a photo gallery, no report here.") == []


# --- team headings, from an audit of 758 real chunks ------------------------------
#
# An audit over Philadelphia's 2025 season showed 10.9% of rows with no team and the
# 49ers missing from the results entirely. Every case below is a heading shape that was
# in the corpus and silently produced nothing.


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Eagles Injury Report", "Eagles"),
        ("Commanders Injury Report", "Commanders"),
        # "49ers" starts with a digit; the old `^[A-Z]` anchor dropped every one
        ("49ers Injury Report", "49ers"),
        # Pittsburgh puts the team in trailing parentheses
        ("Week 17 Injury Report (Browns)", "Browns"),
        ("Wild Card Injury Report (Texans) - Updated", "Texans"),
        # San Francisco appends a qualifier, breaking any end-anchored pattern
        ("Trent Williams Questionable vs. Eagles; Injury Report Ahead of Week 12", "Eagles"),
    ],
)
def test_team_heading_shapes_seen_in_the_corpus(line, expected) -> None:
    assert parse_team_heading(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        # Two clubs means the article title, not a section heading. Treating it as one
        # attributes the first block of rows to whichever side is named first.
        "Eagles at Commanders Injury Report",
        "49ers vs. Eagles Injury Report | Wild Card Round",
        "Thursday's Injury Report",
        "DT Jalen Carter (Shoulders/Did Not Participate)",
        "Did Not Participate",
        "Everyone else fully practiced and is available for Saturday.",
    ],
)
def test_team_heading_rejects_non_headings(line) -> None:
    assert parse_team_heading(line) is None


def test_team_vocabulary_cannot_invent_a_club() -> None:
    """Matching against the 32 real nicknames means a malformed heading yields None
    rather than a plausible-looking team that will never join to anything."""
    assert parse_team_heading("Week 17 Injury Report") is None
    assert parse_team_heading("Final Injury Report (Practice Squad)") is None
