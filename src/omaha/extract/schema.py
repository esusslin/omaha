"""What a valid extracted record looks like, and how to reject one that isn't.

**This module is the reason the extractor is trustworthy, and it contains no LLM call.**
Everything a model returns passes through `validate` before it can reach the database.
Validation is pure, fast, and unit-testable without an API key, which means the part of
the pipeline most likely to be wrong is also the part that's cheapest to test.

Two rules do most of the work:

1. **Closed vocabularies.** `practice_status` is one of three strings or null. Not
   "Limited participation", not "limited", not "LP". Anything else is discarded rather
   than coerced — the same discipline as matching team headings against 32 real
   nicknames instead of a regex, and for the same reason: a value outside the vocabulary
   is a bug, and silently repairing it hides the bug while keeping the bad row.

2. **Grounding.** A player's surname must appear in the chunk the record claims to come
   from. This is the anti-hallucination check, and it's deliberately crude: it cannot
   catch a model that misreads a status, but it does catch the failure that matters most
   — inventing a player who isn't there. Cheap, deterministic, no second model call.

The bias throughout is toward dropping fields rather than guessing them. A missing
practice status shows up honestly in coverage statistics. A wrong one silently corrupts
the feature that `research/practice_signal.py` measured at +0.0297 AUC, and nothing
downstream would ever tell you.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

PRACTICE_STATUSES = ("DNP", "LIMITED", "FULL")
"""Did not participate / limited / full. Mirrors `normalise_practice` in ingest.report,
so table-extracted and prose-extracted rows agree on vocabulary."""

GAME_STATUSES = ("OUT", "DOUBTFUL", "QUESTIONABLE")
"""Null means *not designated*, which is information — the player was on the report and
the club chose not to flag him. Distinct from "we didn't look", which is an absent row."""

REPORT_DAYS = ("WED", "THU", "FRI")
"""Practice reports are filed Wednesday, Thursday and Friday. The trajectory across the
three is the signal; a single flattened status is not."""

MAX_PLAYER_NAME = 128
MAX_INJURY = 64

# Both apostrophe forms: club CMSes emit straight and curly interchangeably, sometimes
# within one page. Written as an escape because the literal trips RUF001.
_SURNAME = re.compile(r"[A-Za-z][A-Za-z'\u2019.-]*")


@dataclass(frozen=True)
class DraftRecord:
    """One candidate fact, before validation. Mirrors the model's JSON exactly."""

    player_name: str
    team: str | None = None
    position: str | None = None
    injury: str | None = None
    practice_status: str | None = None
    game_status: str | None = None
    report_day: str | None = None
    evidence: str | None = None

    @property
    def is_empty(self) -> bool:
        """A record naming a player and asserting nothing about him.

        Worth dropping: it costs a row, adds no fact, and inflates coverage counts with
        entries that would answer no question.
        """
        return not any((self.practice_status, self.game_status, self.injury, self.position))


def _in_vocabulary(value: str | None, allowed: tuple[str, ...]) -> str | None:
    """Uppercase, then accept only exact members. Anything else becomes null.

    Case folding is the one normalisation permitted here, because it's lossless. Mapping
    "Limited participation in practice" onto LIMITED is *not* done — that's the model's
    job, and doing it here would mask a model that isn't following the schema.
    """
    if value is None:
        return None
    candidate = value.strip().upper()
    return candidate if candidate in allowed else None


def surnames(text: str) -> set[str]:
    """Lowercased words from a name or a chunk, for the grounding check."""
    return {w.lower() for w in _SURNAME.findall(text) if len(w) > 1}


def is_grounded(record: DraftRecord, chunk_text: str) -> bool:
    """Does the chunk actually mention this player?

    Any word of the claimed name appearing in the source is enough. Deliberately
    permissive: the goal is catching wholesale invention, not adjudicating whether
    "T.J." matches "TJ". A stricter rule would reject correct records over punctuation,
    and a rejected correct record is indistinguishable, in the coverage stats, from a
    club that filed nothing.
    """
    name_words = surnames(record.player_name)
    if not name_words:
        return False
    return bool(name_words & surnames(chunk_text))


def validate(
    record: DraftRecord, chunk_text: str, *, team_vocabulary: set[str]
) -> DraftRecord | None:
    """Return a cleaned record, or None if it shouldn't be stored at all.

    Dropped entirely when the player isn't named in the source, when the name is
    unusably long or empty, or when nothing is actually asserted. Otherwise individual
    fields are nulled where they fall outside their vocabulary, and the record survives
    carrying whatever it got right — a record with a good practice status and a garbled
    position is still worth having.
    """
    name = record.player_name.strip()
    if not name or len(name) > MAX_PLAYER_NAME:
        return None
    if not is_grounded(replace(record, player_name=name), chunk_text):
        return None

    team = record.team.strip().upper() if record.team else None
    if team not in team_vocabulary:
        team = None

    injury = record.injury.strip() if record.injury else None
    if injury and len(injury) > MAX_INJURY:
        injury = None

    position = record.position.strip().upper() if record.position else None
    if position and (len(position) > 8 or not position.replace("/", "").isalpha()):
        position = None

    cleaned = DraftRecord(
        player_name=name,
        team=team,
        position=position,
        injury=injury,
        practice_status=_in_vocabulary(record.practice_status, PRACTICE_STATUSES),
        game_status=_in_vocabulary(record.game_status, GAME_STATUSES),
        report_day=_in_vocabulary(record.report_day, REPORT_DAYS),
        evidence=(record.evidence or None),
    )
    return None if cleaned.is_empty else cleaned
