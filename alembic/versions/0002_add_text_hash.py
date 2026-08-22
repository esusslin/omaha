"""add text_hash for semantic dedup

Raw-byte hashing is too sensitive: club pages carry build IDs, nonces and rotating
tokens, so identical content produces different bytes on every fetch. Hashing the
*parsed text* instead gives a stable identity for "has anything meaningful changed?"

`content_hash` stays — it identifies the exact bytes we stored, which is what raw file
names and provenance need.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("text_hash", sa.String(64), nullable=True))
    op.create_index("ix_documents_text_hash", "documents", ["source_id", "text_hash"])


def downgrade() -> None:
    op.drop_index("ix_documents_text_hash", table_name="documents")
    op.drop_column("documents", "text_hash")
