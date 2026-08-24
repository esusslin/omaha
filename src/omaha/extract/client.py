"""The one part of extraction that needs an API key.

Deliberately thin. Everything interesting — the prompt, the parser, the validation —
lives in modules that import without credentials, so the pipeline can be tested end to
end offline and this file stays small enough to read in one sitting.

**Absence is handled, not raised.** `available()` reports whether a key is configured,
and the CLI and scheduler check it rather than discovering the problem inside a loop.
Same reasoning as the embedding model degrading to lexical search: a component that
can't run should say so at the top, not fail halfway through a batch having already
spent money.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from omaha.config import get_settings
from omaha.extract.prompt import SYSTEM_PROMPT, build_user_prompt, parse_response
from omaha.extract.schema import DraftRecord

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_TOKENS = 2048
"""Generous for a page of records, bounded enough that a runaway response can't cost
much. A truncated response fails to parse, which leaves the chunk pending for retry."""


class ExtractorUnavailableError(RuntimeError):
    """No API key, or the SDK isn't installed."""


def available() -> bool:
    """Can we extract at all? Checked before a run, not during one."""
    return bool(settings.anthropic_api_key)


@lru_cache(maxsize=1)
def _client() -> Any:
    if not settings.anthropic_api_key:
        raise ExtractorUnavailableError("ANTHROPIC_API_KEY is not set")
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ExtractorUnavailableError("anthropic SDK not installed") from exc
    return Anthropic(api_key=settings.anthropic_api_key)


def extract(chunk_text: str, *, team_hint: str | None = None) -> list[DraftRecord]:
    """One chunk in, candidate records out. Unvalidated — the caller must run `validate`.

    Returning drafts rather than finished records keeps the network boundary and the
    correctness boundary separate: this function can be wrong about what the model said,
    and validation is still the thing that decides what gets stored.

    Temperature 0 because this is extraction, not writing. Two runs over the same chunk
    should agree, otherwise `extractor_version` means nothing and comparing versions
    measures sampling noise.
    """
    response = _client().messages.create(
        model=settings.extract_model,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(chunk_text, team_hint=team_hint)}],
    )

    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    drafts = parse_response(text)
    if not drafts and text.strip():
        # Not an error — plenty of chunks contain no injury facts. Logged at debug so a
        # sudden run of them is findable without drowning a normal run.
        logger.debug("no records parsed from response: %s", text[:200])
    return drafts
