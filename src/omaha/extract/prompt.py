"""The extraction prompt, and the parser for what comes back.

Kept apart from the API client so both halves are testable without a key: the prompt is
a string, the parser is a pure function, and the network call is the only thing that
needs credentials. The parser is where malformed model output has to be survived, and
that's exactly the code you want to exercise offline.

**`EXTRACTOR_VERSION` is the whole iteration story.** It's stamped on every row. Change
the prompt, bump the version, and the corpus becomes unextracted — the hourly job
re-runs it over stored chunks with no refetching, and the old rows stay put until you've
confirmed the new ones are better. Editing a prompt without bumping this produces a
table with two generations of output and no way to tell them apart.
"""

from __future__ import annotations

import json
import re
from typing import Any

from omaha.extract.schema import (
    GAME_STATUSES,
    PRACTICE_STATUSES,
    REPORT_DAYS,
    DraftRecord,
)

EXTRACTOR_VERSION = "v1"
"""Bump on any prompt change. See the module docstring."""

SYSTEM_PROMPT = f"""You extract NFL injury-report facts from text. You return JSON only.

Return an object with one key, "records", whose value is a list. Each element:

  player_name      required, exactly as written in the text
  team             team abbreviation if stated (e.g. PHI, SF), else null
  position         e.g. QB, TE, G/T, else null
  injury           body part or condition, e.g. "Ankle", "Concussion", else null
  practice_status  one of {list(PRACTICE_STATUSES)}, else null
  game_status      one of {list(GAME_STATUSES)}, else null
  report_day       one of {list(REPORT_DAYS)}, else null
  evidence         the shortest span of the source text supporting this record

Rules, in order of importance:

1. Use null for anything the text does not state. Never infer, never guess, never fill a
   field from general knowledge about a player. A null is correct; a plausible invention
   is not.
2. Never emit a player who is not named in the text.
3. Map wording onto the vocabularies above: "did not participate" -> DNP, "limited
   participation" -> LIMITED, "full participation" -> FULL. If the wording does not
   clearly correspond, use null.
4. One record per player per distinct report day. If a passage describes a player on
   Wednesday and again on Friday, that is two records.
5. If the text contains no injury-report facts at all, return {{"records": []}}.
6. Be deterministic. This is extraction, not writing: given the same text twice, return
   the same records in the same order. Do not paraphrase, reword or vary phrasing.

Return only the JSON object. No prose, no code fences."""
"""Rule 6 exists because the SDK removed `temperature` in v1.0 and the documented
replacement for sampling control is a system-prompt instruction. Reproducibility isn't
cosmetic here: `extractor_version` is only meaningful if two runs of the same version
agree, otherwise comparing v1 against v2 measures noise."""


def build_user_prompt(chunk_text: str, *, team_hint: str | None = None) -> str:
    """The per-chunk message.

    `team_hint` carries the team we already know from the document, because a chunk in
    the middle of an article often names players without repeating the club. It's given
    as context rather than as an instruction to fill the field — the model is still told
    to use null when the text doesn't say, and validation checks the value against the
    32-club vocabulary regardless.
    """
    header = f"Source team (context only, may be wrong): {team_hint}\n\n" if team_hint else ""
    return f"{header}Text:\n---\n{chunk_text}\n---"


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_response(raw: str) -> list[DraftRecord]:
    """Model output -> drafts. Returns [] rather than raising, on anything malformed.

    A single unparseable response should cost one chunk, not the run. The chunk stays
    unextracted, so the next pass simply retries it — which is the same resumability
    property that makes embedding safe to interrupt.

    Code fences get stripped even though the prompt forbids them, because "the model
    followed instructions" is not something to build a pipeline on.
    """
    if not raw or not raw.strip():
        return []

    text = _FENCE.sub("", raw).strip()
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, list):
        # Tolerated: a bare list instead of the documented object. Cheap to accept,
        # and rejecting it would discard correct facts over a wrapper.
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("records") or []
    else:
        return []

    if not isinstance(rows, list):
        return []

    drafts: list[DraftRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("player_name")
        if not isinstance(name, str) or not name.strip():
            continue
        drafts.append(
            DraftRecord(
                player_name=name,
                team=_as_str(row.get("team")),
                position=_as_str(row.get("position")),
                injury=_as_str(row.get("injury")),
                practice_status=_as_str(row.get("practice_status")),
                game_status=_as_str(row.get("game_status")),
                report_day=_as_str(row.get("report_day")),
                evidence=_as_str(row.get("evidence")),
            )
        )
    return drafts


def _as_str(value: Any) -> str | None:
    """Coerce a JSON value to a string, treating the many spellings of absence alike.

    Models return null, "null", "" and "N/A" interchangeably for a missing field, and
    the string "null" landing in a database column is a genuinely annoying bug to find
    six weeks later.
    """
    if value is None or not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"null", "none", "n/a", "-", "(-)"}:
        return None
    return cleaned
