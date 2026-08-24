"""Bitemporal document store.

The one rule that matters: never upsert an older snapshot away. Wednesday's injury
report and Friday's are two rows. `knowledge_time` records when *we* learned it, which
is what makes "what did we know at 6pm Wednesday?" answerable — and what keeps any
downstream model honest about leakage.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 768
"""BAAI/bge-base-en-v1.5. Changing this is a migration, not a config tweak."""


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
    """e.g. 'injury_report > table 0 > row 3'"""

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # --- retrieval ---
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    """Stamped per row so re-embedding is additive: write the new version alongside the
    old, switch reads, then drop. Impossible if the vector has no provenance."""
    embedded_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # --- extraction (Phase 4) ---
    extracted_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    """Which extractor has *processed* this chunk — not whether it produced records.

    The distinction matters and costs money if you get it wrong. Plenty of chunks are
    boilerplate, quotes or schedule tables and legitimately yield zero facts. If "has
    records" were the test for "needs extracting", every one of those would be re-sent
    to the API on every hourly run, forever, to rediscover that there's nothing there.

    Same pattern as `embedding_version`: stamped per row, so bumping the extractor makes
    the corpus pending again and leaves the old output in place for comparison."""

    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', text)", persisted=True),
        nullable=True,
    )
    """Lexical half of hybrid retrieval.

    Declared `Computed` so SQLAlchemy knows Postgres owns it and leaves it out of
    INSERTs — the column has existed since migration 0004, but without this the ORM
    couldn't query it, which is why hybrid search needed it added here rather than in a
    new migration. Nothing changes in the database.
    """


class InjuryRecord(Base):
    """A typed fact extracted from a chunk. Phase 4.

    **Why this table exists.** Retrieval returns passages, and a passage is evidence, not
    data. `the-algo` cannot consume "TE Cameron Latu (Ankle) | Practice: LIMITED" — it
    needs a row it can join and feed to a model. This is that row.

    **Derived, never authoritative.** Every record points at the chunk it came from, and
    the chunk points at the document, and the document keeps the original bytes. Nothing
    here is a source of truth; the whole table can be dropped and rebuilt from `chunks`
    without a single HTTP request. That property is the direct lesson of `reparse` —
    extraction frozen at ingest meant improving a parser silently changed nothing, three
    times in a row, before anyone noticed.

    **`knowledge_time` is inherited, never recomputed.** It comes from the parent
    document. An extraction run in March must not make a December fact look like it was
    known in March — that's the leakage the whole store exists to prevent, and it would
    be trivially easy to reintroduce here by stamping `now()`.

    **Every field is nullable on purpose.** The model returns null for anything it can't
    ground in the chunk text. A missing practice status shows up honestly in coverage
    statistics; a hallucinated one silently corrupts the feature that the measured
    +0.0297 AUC result depends on. A wrong value is worse than a missing one, and never
    more so than here.
    """

    __tablename__ = "injury_records"
    __table_args__ = (
        # One record per player per chunk per extractor version. Re-running an unchanged
        # extractor is then a no-op rather than a duplicate, which is what makes the job
        # safe to fire hourly.
        UniqueConstraint(
            "chunk_id", "player_name", "extractor_version", name="uq_record_chunk_player"
        ),
        Index("ix_records_team_knowledge", "team", "knowledge_time"),
        Index("ix_records_player", "player_name"),
        # The query the scheduler runs every hour: what hasn't been extracted yet?
        Index("ix_records_version", "extractor_version"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"))
    chunk: Mapped[Chunk] = relationship()
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    """Denormalised from the chunk so team/date queries don't need a three-table join."""

    # --- the fact ---
    player_name: Mapped[str] = mapped_column(String(128))
    """As written in the source. Normalisation is a separate concern with its own
    failure modes, and conflating the two makes both harder to measure."""
    player_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    """Resolved identity, once a crosswalk exists. Null until then — and null is honest:
    "T.J. Watt", "TJ Watt" and "Watt, T.J." are one player and guessing which costs more
    than admitting we haven't joined them yet."""

    team: Mapped[str | None] = mapped_column(String(8), nullable=True)
    position: Mapped[str | None] = mapped_column(String(8), nullable=True)
    injury: Mapped[str | None] = mapped_column(String(64), nullable=True)

    practice_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    """DNP | LIMITED | FULL. Validated against a closed set before insert — the same
    discipline as team headings, for the same reason: a value outside the vocabulary is
    a bug, not a discovery."""
    game_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    """OUT | DOUBTFUL | QUESTIONABLE | null. Null means not designated, which is itself
    information and must not be confused with "we didn't look"."""
    report_day: Mapped[str | None] = mapped_column(String(16), nullable=True)
    """Wed | Thu | Fri. The trajectory across these three is the signal; a single
    flattened status is not."""

    # --- provenance ---
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    """The span of source text supporting this record. What lets an agent cite rather
    than assert, and what makes a hallucination checkable by a human in one glance."""

    knowledge_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    """Inherited from the document. See the class docstring."""
    published_time: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    extractor_version: Mapped[str] = mapped_column(String(32))
    """Which prompt produced this. Bump it and the corpus becomes unextracted again, so
    a prompt change is a re-run rather than a migration — and the old rows survive long
    enough to answer "did the new one actually do better?"."""
    extracted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class JobRun(Base):
    """One execution of a scheduled job.

    A collector you can't observe is a collector you'll discover was broken in week
    six. "Did Wednesday's sweep run, and what did it find?" should be a query.
    """

    __tablename__ = "job_runs"
    __table_args__ = (Index("ix_job_runs_name_started", "job_name", "started_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job_name: Mapped[str] = mapped_column(String(64))

    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ok: Mapped[bool] = mapped_column(Boolean, default=False)

    sources_attempted: Mapped[int] = mapped_column(Integer, default=0)
    sources_failed: Mapped[int] = mapped_column(Integer, default=0)
    documents_created: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()
