"""Bitemporal writes.

The rule from `the-algo`: never upsert an older snapshot away. Wednesday's report and
Friday's are separate rows, so "what did we know at 6pm Wednesday?" stays answerable.

The refinement that keeps it from exploding: **a new row is written only when the
content actually changes.** Polling hourly against unchanged content would otherwise
produce 24 identical rows a day. So `knowledge_time` means *when we first saw this
version* — which is exactly the moment that matters for leakage — and reconstructing
what we knew at time T is "the latest document for this source with
knowledge_time <= T".

**Dedup is on parsed text, not raw bytes.** Club pages carry build IDs, nonces and
rotating tokens, so the bytes differ on every single fetch while the actual report is
unchanged. Byte hashing produced a new row per poll, which defeats the point. The
parsed text is stable, so that's the identity that decides "is this new?"

Unchanged content still updates the source's health, because confirming nothing changed
is a successful poll.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from omaha.config import get_settings
from omaha.db.models import Document, Source
from omaha.ingest.fetch import FetchResult
from omaha.ingest.parse import ParsedDocument

settings = get_settings()


_WHITESPACE = re.compile(r"\s+")


def text_fingerprint(text: str) -> str:
    """Stable identity for parsed content.

    Whitespace is collapsed before hashing so a reflow or an extra blank line doesn't
    read as a change. Anything beyond that — a word differing — is a real change and
    should produce a new snapshot.
    """
    return hashlib.sha256(_WHITESPACE.sub(" ", text).strip().encode("utf-8")).hexdigest()


def _raw_path(source: Source, fetched_at: dt.datetime, content_hash: str) -> Path:
    """Where the untouched original lives. Kept so parsers can be re-run later."""
    day = fetched_at.strftime("%Y/%m/%d")
    root = Path(settings.data_dir) / "raw" / source.kind / day
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{source.name}-{content_hash[:12]}"


def latest_document(session: Session, source: Source) -> Document | None:
    return session.scalars(
        select(Document)
        .where(Document.source_id == source.id)
        .order_by(desc(Document.knowledge_time))
        .limit(1)
    ).first()


def as_of(session: Session, source: Source, when: dt.datetime) -> Document | None:
    """What we knew from this source at `when`. The whole point of the schema."""
    return session.scalars(
        select(Document)
        .where(Document.source_id == source.id, Document.knowledge_time <= when)
        .order_by(desc(Document.knowledge_time))
        .limit(1)
    ).first()


def record_failure(session: Session, source: Source, error: str, at: dt.datetime) -> None:
    source.last_attempt_at = at
    source.last_error = error
    source.consecutive_failures += 1
    session.flush()


def record_unchanged(session: Session, source: Source, at: dt.datetime) -> None:
    """A 304, or identical bytes. Nothing to store; the source is healthy."""
    source.last_attempt_at = at
    source.last_success_at = at
    source.last_error = None
    source.consecutive_failures = 0
    session.flush()


def store_document(
    session: Session,
    source: Source,
    fetch_result: FetchResult,
    parsed: ParsedDocument,
    *,
    season: int | None = None,
    week: int | None = None,
) -> Document | None:
    """Persist a fetched document if — and only if — its content is new.

    Returns the new Document, or None when content was unchanged.
    """
    at = fetch_result.fetched_at
    content_hash = fetch_result.content_hash
    if content_hash is None or fetch_result.content is None:
        record_failure(session, source, "no content to store", at)
        return None

    text_hash = text_fingerprint(parsed.text)

    previous = latest_document(session, source)
    if previous is not None and previous.text_hash == text_hash:
        # Bytes may differ — build IDs, nonces — but nothing meaningful changed.
        record_unchanged(session, source, at)
        return None

    raw_ref = _raw_path(source, at, content_hash)
    raw_ref.write_bytes(fetch_result.content)

    document = Document(
        source_id=source.id,
        source_url=fetch_result.url,
        doc_type=source.kind,
        team=source.team,
        season=season,
        week=week,
        # knowledge_time is when WE learned it. Never backdated, never taken from the
        # document's own claimed publication date — that's `published_time`.
        knowledge_time=at,
        published_time=None,
        fetch_time=at,
        content_hash=content_hash,
        text_hash=text_hash,
        raw_ref=str(raw_ref),
        parsed_text=parsed.text or None,
        parsed_tables={"tables": parsed.tables, "parser": parsed.parser} if parsed.tables else None,
    )
    session.add(document)

    source.last_attempt_at = at
    source.last_success_at = at
    source.last_error = None
    source.consecutive_failures = 0
    if fetch_result.etag:
        source.etag = fetch_result.etag
    if fetch_result.last_modified:
        source.last_modified = fetch_result.last_modified

    session.flush()
    return document
