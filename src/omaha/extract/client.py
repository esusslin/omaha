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

import datetime as dt
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


KEY_PREFIX = "sk-ant-"
MIN_KEY_LENGTH = 40


def available() -> bool:
    """Can we extract at all? Checked before a run, not during one.

    Shape-checked rather than merely non-empty, because the first version of this
    reported "present" for the literal string `sk-ant-...` pasted from a README. A
    readiness check that passes on a placeholder is worse than no check: it moves the
    failure from startup, where it's obvious, into the middle of a loop, where it looks
    like an API problem.

    This can't tell a revoked key from a live one — only the API can — but it catches
    every version of "I meant to fill that in".
    """
    key = settings.anthropic_api_key.strip()
    return key.startswith(KEY_PREFIX) and len(key) >= MIN_KEY_LENGTH and "..." not in key


@lru_cache(maxsize=1)
def _client() -> Any:
    if not settings.anthropic_api_key:
        raise ExtractorUnavailableError("ANTHROPIC_API_KEY is not set")
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ExtractorUnavailableError("anthropic SDK not installed") from exc
    return Anthropic(api_key=settings.anthropic_api_key)


def extract(
    chunk_text: str,
    *,
    team_hint: str | None = None,
    published: dt.datetime | None = None,
) -> list[DraftRecord]:
    """One chunk in, candidate records out. Unvalidated — the caller must run `validate`.

    Returning drafts rather than finished records keeps the network boundary and the
    correctness boundary separate: this function can be wrong about what the model said,
    and validation is still the thing that decides what gets stored.

    **No `temperature`.** This wanted temperature 0 — extraction should be reproducible,
    or `extractor_version` means nothing and comparing two versions measures sampling
    noise. But the anthropic SDK removed `temperature`, `top_p` and `top_k` from the
    Messages methods in v1.0, and the documented replacement is to ask for the behaviour
    in the system prompt, which `SYSTEM_PROMPT` now does. Passing it raises TypeError
    before any request is made, so this fails loudly rather than silently sampling.
    """
    response = _client().messages.create(
        model=settings.extract_model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": build_user_prompt(chunk_text, team_hint=team_hint, published=published),
            }
        ],
    )

    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    drafts = parse_response(text)
    if not drafts and text.strip():
        # Not an error — plenty of chunks contain no injury facts. Logged at debug so a
        # sudden run of them is findable without drowning a normal run.
        logger.debug("no records parsed from response: %s", text[:200])
    return drafts
