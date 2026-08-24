"""mark chunks as processed by the extractor

Separate from 0005 on purpose. `extracted_version` records that a chunk has been *seen*
by an extractor, which is not the same as it having produced records — plenty of chunks
are quotes, boilerplate or schedule tables and correctly yield nothing. If "has records"
were the test for "needs extracting", every one of those would be re-sent to the API on
every run, forever, to rediscover that there is nothing in it.

Same pattern as `embedding_version` on the same table: stamped per row, so bumping the
extractor makes the corpus pending again while leaving the previous output in place for
comparison.

(It lives in its own revision rather than in 0005 because 0005 had already been applied
when the need for it surfaced. Editing an applied migration leaves every database that
ran the old version permanently out of step with the file — and the failure shows up as
a downgrade that tries to drop something that was never created.)

Revision ID: 0006
Revises: 0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("extracted_version", sa.String(32), nullable=True))
    # The query the scheduler runs every hour: which chunks still need extracting?
    op.create_index("ix_chunks_unextracted", "chunks", ["extracted_version"])


def downgrade() -> None:
    op.drop_index("ix_chunks_unextracted", table_name="chunks")
    op.drop_column("chunks", "extracted_version")
