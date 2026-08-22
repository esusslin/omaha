"""Bitemporal document store.

The one rule that matters: never upsert an older snapshot away. Wednesday's injury
report and Friday's are two rows. `knowledge_time` records when *we* learned it, which
is what makes "what did we know at 6pm Wednesday?" answerable — and what keeps any
downstream model honest about leakage.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Source(Base):
    """A place we fetch from, and whether it is currently healthy."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    kind: Mapped[str] = mapped_column(String(32))
    """injury_report | inactives | transactions | transcript | depth_chart | gamebook | rss"""

    url: Mapped[str] = mapped_column(Text)
    team: Mapped[str | None] = mapped_column(String(8), nullable=True)

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    cadence_seconds: Mapped[int] = mapped_column(Integer, default=3600)

    last_success_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_attempt_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)

    # conditional-request caching so we are polite to origins
    etag: Mapped[str | None] = mapped_column(String(256), nullable=True)
    last_modified: Mapped[str | None] = mapped_column(String(128), nullable=True)

    documents: Mapped[list[Document]] = relationship(back_populates="source")

    @property
    def is_stale(self) -> bool:
        if not self.enabled:
            return False
        if self.last_success_at is None:
            return True
        age = (_utcnow() - self.last_success_at).total_seconds()
        return age > self.cadence_seconds * 2


class Document(Base):
    """One fetched artifact, at one moment. Immutable once written."""

    __tablename__ = "documents"
    __table_args__ = (
        # the same bytes from the same source at the same knowledge_time is a no-op,
        # but the same bytes at a LATER knowledge_time is a new row on purpose
        UniqueConstraint(
            "source_id", "content_hash", "knowledge_time", name="uq_document_snapshot"
        ),
        Index("ix_documents_knowledge_time", "knowledge_time"),
        Index("ix_documents_team_week", "team", "season", "week"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    source: Mapped[Source] = relationship(back_populates="documents")

    source_url: Mapped[str] = mapped_column(Text)
    doc_type: Mapped[str] = mapped_column(String(32))

    team: Mapped[str | None] = mapped_column(String(8), nullable=True)
    season: Mapped[int | None] = mapped_column(Integer, nullable=True)
    week: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- the bitemporal triple ---
    knowledge_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    """When we learned it. Never backdate this."""
    published_time: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    """When the source claims it was published, if it says."""
    fetch_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    """When the HTTP request happened."""

    content_hash: Mapped[str] = mapped_column(String(64))
    """SHA-256 of the raw bytes. Identifies exactly what we stored."""
    text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    """SHA-256 of the parsed text — what we dedup on.

    Raw bytes are too sensitive: club pages embed build IDs, nonces and rotating tokens,
    so byte-identical content is rare even when nothing meaningful changed. The parsed
    text is stable, so it's the right identity for "did this actually change?"
    """
    raw_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Path to the untouched original."""

    parsed_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_tables: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    chunks: Mapped[list[Chunk]] = relationship(back_populates="document")


class Chunk(Base):
    """A retrievable unit that remembers where it came from.

    Span offsets are kept so every downstream claim can cite exact source text.
    The embedding column arrives in Phase 2 — deliberately absent here so Day 1
    doesn't depend on a model download.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunk_ordinal"),
        Index("ix_chunks_document", "document_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    document: Mapped[Document] = relationship(back_populates="chunks")

    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)

    span_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    span_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    """e.g. 'injury_report > Wednesday > offense'"""

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
