"""Scheduled collection.

**Timezone matters here.** NFL policy requires practice reports by 4:00 pm Eastern on
Wednesday, Thursday and Friday. So the injury sweep is cron'd in `America/New_York`,
not UTC — otherwise it drifts an hour twice a year and misses the window in November,
which is exactly when the reports start mattering.

We poll at 17:00 ET, an hour after the deadline, so late filers are captured.

Everything is idempotent: the sweep gates on each source's cadence, and the store only
writes when content actually changed. A double-fire costs a conditional request.
"""

from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from omaha.db.session import session_scope
from omaha.ingest.sweep import sweep

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")


def _run(job_name: str, kind: str | None) -> None:
    """Run a sweep in its own session, logging rather than raising.

    A scheduler job that raises kills nothing but its own next run — but it also loses
    the outcome, so we catch, log, and let the JobRun row carry the detail.
    """
    try:
        with session_scope() as session:
            outcome = sweep(session, job_name=job_name, kind=kind, due_only=True)
        logger.info(
            "sweep complete job=%s attempted=%d failed=%d created=%d",
            job_name,
            outcome.attempted,
            outcome.failed,
            outcome.created,
        )
        for line in outcome.lines or []:
            logger.info("  %s", line)
    except Exception:
        logger.exception("sweep failed job=%s", job_name)


def _index() -> None:
    """Chunk anything new, then embed it if the model is present.

    Without this the collector is only half a pipeline: documents land in the store and
    stay invisible to search, because chunking and embedding were manual `make` targets.
    Ingestion that doesn't reach the index is ingestion nobody can query.

    Embedding is skipped where `fastembed` isn't installed — the host — so the same
    scheduler runs on a laptop and in the Linux container, doing as much as it can in
    each. Both steps are idempotent and resumable, so a partial run costs nothing.
    """
    from sqlalchemy import select

    from omaha.db.models import Chunk, Document
    from omaha.retrieve.chunk import chunk_document

    try:
        with session_scope() as session:
            chunked = select(Chunk.document_id).distinct().scalar_subquery()
            documents = session.scalars(
                select(Document).where(Document.id.not_in(chunked)).limit(500)
            ).all()

            made = 0
            for document in documents:
                tables = (document.parsed_tables or {}).get("tables")
                for draft in chunk_document(
                    doc_type=document.doc_type,
                    text=document.parsed_text or "",
                    tables=tables,
                ):
                    session.add(
                        Chunk(
                            document_id=document.id,
                            ordinal=draft.ordinal,
                            text=draft.text,
                            section_path=draft.section_path,
                            span_start=draft.span_start,
                            span_end=draft.span_end,
                        )
                    )
                    made += 1
            if made:
                logger.info("indexed %d chunks from %d documents", made, len(documents))
    except Exception:
        logger.exception("chunking failed")
        return

    try:
        from omaha.retrieve.embed import embed_all
    except Exception as exc:
        logger.debug("no embedding model (%s) — chunks left unembedded", type(exc).__name__)
        return

    try:
        with session_scope() as session:
            count = embed_all(session)
        if count:
            logger.info("embedded %d chunks", count)
    except Exception:
        logger.exception("embedding failed")


def build_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=EASTERN)

    # Practice reports: Wed/Thu/Fri at 17:00 ET, an hour after the league deadline.
    # Both kinds: `injury_index` sources discover article URLs, `injury_report` covers
    # any direct source still registered.
    scheduler.add_job(
        _run,
        CronTrigger(day_of_week="wed,thu,fri", hour=17, minute=0, timezone=EASTERN),
        args=["injury_sweep", ["injury_index", "injury_report"]],
        id="injury_sweep",
        max_instances=1,
        coalesce=True,  # a missed fire runs once, not N times
        misfire_grace_time=3600,
    )

    # Close the loop: fetched documents are useless until they're chunked and embedded.
    # Offset from the sweeps so it runs against what they just wrote.
    scheduler.add_job(
        _index,
        CronTrigger(minute=25, timezone=EASTERN),
        id="index",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=1800,
    )

    # Transactions and news: hourly. Cadence gating inside the sweep does the throttling.
    scheduler.add_job(
        _run,
        CronTrigger(minute=7, timezone=EASTERN),
        args=["hourly_sweep", None],
        id="hourly_sweep",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=1800,
    )

    return scheduler
