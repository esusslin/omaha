"""Turning validated drafts into rows, and choosing which chunks to process.

The two functions here are where the cost control and the leakage control live.

`pending_chunks` decides what gets sent to the API. It selects on `extracted_version`,
not on "has records" — see the column's docstring for why that distinction is worth a
migration.

`persist` is the only place records are written, and it is the only place
`knowledge_time` is set. It always copies from the parent document and never calls
`now()`. That single line is the difference between a corpus you can backtest against
and one where every fact appears to have been known the moment you happened to run the
extractor.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence

from sqlalchemy import case, select
from sqlalchemy.orm import Session

from omaha.db.models import Chunk, Document, InjuryRecord
from omaha.extract.schema import DraftRecord, validate
from omaha.ingest.seeds import CLUBS

logger = logging.getLogger(__name__)

TEAM_VOCABULARY = {club.abbr for club in CLUBS}
"""The 32 real abbreviations. A team outside this set is a model error, not a discovery."""


def pending_chunks(session: Session, version: str, *, limit: int = 50) -> Sequence[Chunk]:
    """Chunks not yet processed by this extractor version, most valuable first.

    **Ordering is a cost decision, not a cosmetic one.** Every run is bounded by a batch
    size, so whatever sorts first is what the budget gets spent on. Ordering by id alone
    spent the first five calls on a practice-report *legend* and a podcast promo — all
    correctly returning nothing, all costing the same as a real row.

    `injury_report` documents first, then everything else, then by id within each group
    so a backfill still makes monotonic progress and an interrupted run resumes rather
    than re-drawing a random sample.
    """
    priority = case((Document.doc_type == "injury_report", 0), else_=1)
    statement = (
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where((Chunk.extracted_version.is_(None)) | (Chunk.extracted_version != version))
        .order_by(priority, Chunk.id)
        .limit(limit)
    )
    return session.scalars(statement).all()


def pending_count(session: Session, version: str) -> int:
    from sqlalchemy import func

    return (
        session.scalar(
            select(func.count(Chunk.id)).where(
                (Chunk.extracted_version.is_(None)) | (Chunk.extracted_version != version)
            )
        )
        or 0
    )


def persist(
    session: Session,
    chunk: Chunk,
    drafts: Sequence[DraftRecord],
    *,
    version: str,
) -> int:
    """Validate, store, and mark the chunk processed. Returns rows written.

    The chunk is stamped whether or not anything survived validation — a chunk that
    genuinely contains no injury facts is *done*, not pending, and treating it otherwise
    means paying to rediscover that on every run.
    """
    document = session.get(Document, chunk.document_id)
    if document is None:  # pragma: no cover - FK makes this unreachable
        logger.warning("chunk %s has no document", chunk.id)
        return 0

    existing = {
        name
        for (name,) in session.execute(
            select(InjuryRecord.player_name).where(
                InjuryRecord.chunk_id == chunk.id,
                InjuryRecord.extractor_version == version,
            )
        )
    }

    written = 0
    for draft in drafts:
        record = validate(draft, chunk.text, team_vocabulary=TEAM_VOCABULARY)
        if record is None:
            continue
        if record.player_name in existing:
            # The unique constraint would reject this anyway; catching it here keeps a
            # re-run from aborting the surrounding transaction over an expected collision.
            continue
        existing.add(record.player_name)

        session.add(
            InjuryRecord(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                player_name=record.player_name,
                team=record.team or document.team,
                position=record.position,
                injury=record.injury,
                practice_status=record.practice_status,
                game_status=record.game_status,
                report_day=record.report_day,
                evidence=record.evidence,
                # Inherited, never recomputed. An extraction run today must not make a
                # December fact look like it was known today.
                knowledge_time=document.knowledge_time,
                published_time=document.published_time,
                extractor_version=version,
                extracted_at=dt.datetime.now(dt.UTC),
            )
        )
        written += 1

    chunk.extracted_version = version
    return written
