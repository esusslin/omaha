"""initial bitemporal document store

Revision ID: 0001
Revises:
Create Date: 2026-08-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("team", sa.String(8), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("cadence_seconds", sa.Integer(), nullable=False, server_default="3600"),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("etag", sa.String(256), nullable=True),
        sa.Column("last_modified", sa.String(128), nullable=True),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("sources.id"), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("doc_type", sa.String(32), nullable=False),
        sa.Column("team", sa.String(8), nullable=True),
        sa.Column("season", sa.Integer(), nullable=True),
        sa.Column("week", sa.Integer(), nullable=True),
        # the bitemporal triple
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fetch_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("raw_ref", sa.Text(), nullable=True),
        sa.Column("parsed_text", sa.Text(), nullable=True),
        sa.Column("parsed_tables", postgresql.JSONB(), nullable=True),
        sa.UniqueConstraint(
            "source_id", "content_hash", "knowledge_time", name="uq_document_snapshot"
        ),
    )
    op.create_index("ix_documents_knowledge_time", "documents", ["knowledge_time"])
    op.create_index("ix_documents_team_week", "documents", ["team", "season", "week"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("span_start", sa.Integer(), nullable=True),
        sa.Column("span_end", sa.Integer(), nullable=True),
        sa.Column("section_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_chunk_ordinal"),
    )
    op.create_index("ix_chunks_document", "chunks", ["document_id"])


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("sources")
