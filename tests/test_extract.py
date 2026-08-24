"""Extraction validation and response parsing. No API key, no database.

These are the tests that matter most in Phase 4, because the LLM call is the part of the
system that can be confidently, fluently wrong. Everything a model returns has to
survive `validate` before it reaches a table, so this file is where the guarantees live.
"""

from __future__ import annotations

import pytest

from omaha.extract.prompt import (
    EXTRACTOR_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_response,
)
from omaha.extract.schema import DraftRecord, is_grounded, validate

TEAMS = {"PHI", "SF", "PIT", "ATL", "WAS"}

CHUNK = (
    "TE Cameron Latu (Ankle) | Practice: LIMITED | Injury: Ankle\n"
    "T Trey Pipkins (Back) | Practice: DNP | Injury: Back"
)


def clean(record: DraftRecord, text: str = CHUNK) -> DraftRecord | None:
    return validate(record, text, team_vocabulary=TEAMS)


# --- grounding: the anti-hallucination check ----------------------------------------


def test_a_player_absent_from_the_source_is_dropped() -> None:
    """The failure that matters most. A model asked about injuries will happily produce a
    plausible NFL player who is nowhere in the text, and a fluent invention is far more
    dangerous than a blank — it looks exactly like a real record downstream."""
    assert clean(DraftRecord(player_name="Patrick Mahomes", practice_status="DNP")) is None


def test_a_player_present_in_the_source_survives() -> None:
    record = clean(DraftRecord(player_name="Cameron Latu", practice_status="LIMITED"))
    assert record is not None
    assert record.practice_status == "LIMITED"


def test_grounding_matches_on_any_name_word() -> None:
    """Deliberately permissive: a surname is enough. Being stricter would reject correct
    records over punctuation, and a rejected correct record is indistinguishable from a
    club that filed nothing."""
    assert is_grounded(DraftRecord(player_name="C.J. Latu"), CHUNK)
    assert not is_grounded(DraftRecord(player_name="Justin Jefferson"), CHUNK)


def test_an_empty_name_is_dropped() -> None:
    assert clean(DraftRecord(player_name="   ", practice_status="FULL")) is None


# --- closed vocabularies ------------------------------------------------------------


def test_out_of_vocabulary_practice_status_becomes_null_not_a_guess() -> None:
    """ "Limited participation in practice" is the *source* wording, and mapping it here
    would hide a model that isn't following the schema. Null keeps the failure visible in
    coverage statistics instead of burying it in a column."""
    record = clean(
        DraftRecord(
            player_name="Latu",
            practice_status="Limited participation in practice",
            game_status="QUESTIONABLE",
        )
    )
    assert record is not None
    assert record.practice_status is None
    assert record.game_status == "QUESTIONABLE"


def test_vocabulary_matching_is_case_insensitive() -> None:
    """Case folding is lossless, so it's the one normalisation allowed."""
    record = clean(DraftRecord(player_name="Latu", practice_status="dnp", report_day="wed"))
    assert record is not None
    assert record.practice_status == "DNP"
    assert record.report_day == "WED"


def test_a_team_outside_the_league_is_discarded() -> None:
    record = clean(DraftRecord(player_name="Latu", team="XYZ", practice_status="FULL"))
    assert record is not None
    assert record.team is None


def test_a_real_team_is_kept() -> None:
    record = clean(DraftRecord(player_name="Latu", team="sf", practice_status="FULL"))
    assert record is not None
    assert record.team == "SF"


def test_one_bad_field_does_not_discard_the_good_ones() -> None:
    """A record with a valid practice status and a garbled position is still worth
    having. Dropping the row would lose a fact to punish a typo."""
    record = clean(
        DraftRecord(
            player_name="Pipkins",
            position="left tackle, probably",
            practice_status="DNP",
        )
    )
    assert record is not None
    assert record.position is None
    assert record.practice_status == "DNP"


def test_a_record_asserting_nothing_is_dropped() -> None:
    """Names a player, states no fact. Costs a row, answers no question, and inflates
    coverage counts with entries that look like data."""
    assert clean(DraftRecord(player_name="Cameron Latu")) is None


# --- response parsing ---------------------------------------------------------------


def test_parses_the_documented_shape() -> None:
    raw = '{"records": [{"player_name": "Cameron Latu", "practice_status": "LIMITED"}]}'
    drafts = parse_response(raw)
    assert len(drafts) == 1
    assert drafts[0].player_name == "Cameron Latu"


def test_survives_code_fences_the_prompt_forbade() -> None:
    """The prompt says no fences. "The model followed instructions" is not a thing to
    build a pipeline on."""
    raw = '```json\n{"records": [{"player_name": "Latu", "practice_status": "FULL"}]}\n```'
    assert len(parse_response(raw)) == 1


def test_accepts_a_bare_list() -> None:
    raw = '[{"player_name": "Latu", "practice_status": "FULL"}]'
    assert len(parse_response(raw)) == 1


def test_malformed_json_costs_one_chunk_not_the_run() -> None:
    """Returning [] leaves the chunk unextracted, so the next pass retries it. Raising
    would abort a batch over one bad response."""
    assert parse_response("I'm sorry, I can't help with that.") == []
    assert parse_response('{"records": [') == []
    assert parse_response("") == []


def test_the_many_spellings_of_absence_all_become_none() -> None:
    """Models return null, "null", "", "N/A" and "(-)" interchangeably. The literal
    string "null" reaching a database column is a genuinely annoying bug to find later."""
    raw = (
        '{"records": [{"player_name": "Latu", "team": "null", "injury": "N/A", '
        '"position": "", "game_status": "(-)"}]}'
    )
    draft = parse_response(raw)[0]
    assert draft.team is None
    assert draft.injury is None
    assert draft.position is None
    assert draft.game_status is None


def test_rows_without_a_player_name_are_skipped() -> None:
    raw = '{"records": [{"practice_status": "DNP"}, {"player_name": "Latu"}]}'
    assert [d.player_name for d in parse_response(raw)] == ["Latu"]


# --- prompt -------------------------------------------------------------------------


def test_team_hint_is_context_not_instruction() -> None:
    """The hint helps with chunks that name players without repeating the club, but the
    model is still told to use null when the text doesn't say — and validation checks the
    result against the league vocabulary regardless."""
    prompt = build_user_prompt("some text", team_hint="PHI")
    assert "PHI" in prompt
    assert "may be wrong" in prompt


def test_prompt_omits_the_hint_when_there_is_none() -> None:
    assert build_user_prompt("some text").startswith("Text:")


def test_extractor_version_is_set() -> None:
    """Stamped on every row. Without it the table holds two generations of output with
    no way to tell them apart, and "did v2 beat v1?" becomes unanswerable."""
    assert EXTRACTOR_VERSION


def test_the_prompt_asks_for_determinism() -> None:
    """The SDK removed `temperature` in v1.0, so sampling control lives in the prompt
    now. Reproducibility isn't cosmetic: comparing extractor versions is meaningless if
    two runs of the *same* version disagree."""
    assert "deterministic" in SYSTEM_PROMPT.lower()


# --- readiness ----------------------------------------------------------------------


def test_a_placeholder_key_does_not_report_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """`available()` once returned True for the literal string pasted out of a README.

    A readiness check that passes on a placeholder is worse than none — it moves the
    failure from startup, where it's obvious, into the middle of a loop, where it looks
    like an API outage."""
    from omaha.extract import client

    for placeholder in ("", "   ", "sk-ant-...", "your-key-here", "sk-ant-short"):
        monkeypatch.setattr(client.settings, "anthropic_api_key", placeholder)
        assert not client.available(), f"{placeholder!r} should not report ready"


def test_a_plausible_key_reports_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    from omaha.extract import client

    monkeypatch.setattr(client.settings, "anthropic_api_key", "sk-ant-" + "a" * 60)
    assert client.available()
