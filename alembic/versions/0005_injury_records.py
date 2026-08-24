"""typed injury records extracted from chunks

Phase 4. Retrieval returns passages; models need rows. This is the table that turns one
into the other.

Everything here is derived — droppable and rebuildable from `chunks` with no network
access — which is why the foreign keys cascade and why nothing in it is authoritative.

`extractor_version` is stamped per row for the same reason `embedding_version` is on
chunks: a prompt change should be additive. Write the new version alongside the old,
compare, then delete the loser. Overwriting in place makes "is v2 better than v1?"
unanswerable, and that question is the whole point of having an eval harness.

Revision ID: 0005
Revises: 0004
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "injury_records",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "chunk_id",
            sa.BigInteger(),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("player_name", sa.String(128), nullable=False),
        sa.Column("player_id", sa.String(32), nullable=True),
        sa.Column("team", sa.String(8), nullable=True),
        sa.Column("position", sa.String(8), nullable=True),
        sa.Column("injury", sa.String(64), nullable=True),
        sa.Column("practice_status", sa.String(16), nullable=True),
        sa.Column("game_status", sa.String(16), nullable=True),
        sa.Column("report_day", sa.String(16), nullable=True),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extractor_version", sa.String(32), nullable=False),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Re-running an unchanged extractor must be a no-op, not a duplicate — the hourly
    # job depends on it.
    op.create_unique_constraint(
        "uq_record_chunk_player",
        "injury_records",
        ["chunk_id", "player_name", "extractor_version"],
    )

    # "What did we know about this team at this moment?" — the query the-algo asks.
    op.create_index("ix_records_team_knowledge", "injury_records", ["team", "knowledge_time"])
    op.create_index("ix_records_player", "injury_records", ["player_name"])
    # "What hasn't been extracted yet?" — the query the scheduler asks hourly.
    op.create_index("ix_records_version", "injury_records", ["extractor_version"])


def downgrade() -> None:
    op.drop_index("ix_records_version", table_name="injury_records")
    op.drop_index("ix_records_player", table_name="injury_records")
    op.drop_index("ix_records_team_knowledge", table_name="injury_records")
    op.drop_constraint("uq_record_chunk_player", "injury_records", type_="unique")
    op.drop_table("injury_records")
