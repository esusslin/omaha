"""add embeddings and full-text to chunks

768 dimensions — BAAI/bge-base-en-v1.5 via ONNX. Model and version are stored per row so
re-embedding is an additive migration rather than a wipe: write new rows with the new
version, switch reads, drop the old. You cannot do that if the vector column has no
provenance.

The tsvector is a generated column, so lexical search stays in sync with the text
automatically. Hybrid retrieval (Day 5) needs both halves.

Revision ID: 0004
Revises: 0003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "chunks", sa.Column("embedding", sa.dialects.postgresql.ARRAY(sa.Float()), nullable=True)
    )
    op.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM}) USING NULL")

    op.add_column("chunks", sa.Column("embedding_model", sa.String(128), nullable=True))
    op.add_column("chunks", sa.Column("embedding_version", sa.String(32), nullable=True))
    op.add_column("chunks", sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True))

    # lexical half of hybrid retrieval — generated, so it can't drift from `text`
    op.execute(
        "ALTER TABLE chunks ADD COLUMN tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING GIN (tsv)")

    # HNSW for cosine. Built now while the table is small; building it on a full corpus
    # is slow and locks writes.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.create_index("ix_chunks_unembedded", "chunks", ["embedding_version"])


def downgrade() -> None:
    op.drop_index("ix_chunks_unembedded", table_name="chunks")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_chunks_tsv")
    op.drop_column("chunks", "tsv")
    op.drop_column("chunks", "embedded_at")
    op.drop_column("chunks", "embedding_version")
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "embedding")
