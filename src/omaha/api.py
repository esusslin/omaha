"""FastAPI application.

/health reports per-source staleness and recent job outcomes rather than just process
liveness. It returns 200 while the process is alive — so the platform doesn't restart a
healthy container over stale upstream data — and sets ok=false when a source is overdue
or the last scheduled sweep failed. Point an external pinger at it and alert on `ok`,
not on status code.

(Pattern lifted from the-algo, which learned it the hard way.)
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from omaha.config import get_settings
from omaha.db.models import Document, JobRun, Source
from omaha.db.session import get_session
from omaha.scheduler import build_scheduler
from omaha.search_api import router as search_router

logger = logging.getLogger(__name__)
settings = get_settings()


def configure_logging() -> None:
    """Attach a handler to the root logger, honouring `log_level`.

    Without this every `logger.info` in the collector goes nowhere: Python's root logger
    defaults to WARNING with no handler, and uvicorn only configures its own loggers.
    Locally that's invisible because the CLI prints as it goes — but in a container the
    scheduler is the only thing running, and it was reporting neither success nor
    failure. A collector nobody can observe is indistinguishable from one that isn't
    running, which is the same class of problem as a healthy pipeline over an empty pipe.

    `force=True` because uvicorn may have configured handlers first; without it this is
    a silent no-op, which would be a fitting bug for a logging fix to have.
    """
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )
    # APScheduler logs every job submission at INFO. Useful when a job misfires,
    # noise otherwise.
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)


configure_logging()

# Annotated dependency alias — keeps Depends() out of argument defaults (ruff B008)
DbSession = Annotated[Session, Depends(get_session)]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Start the collector alongside the API, unless explicitly disabled.

    Set OMAHA_SCHEDULER_ENABLED=false when running multiple replicas — two schedulers
    against one database means duplicate fetches. Single replica is the intended shape
    for now.
    """
    scheduler = None
    if settings.scheduler_enabled:
        scheduler = build_scheduler()
        scheduler.start()
        logger.info("scheduler started with jobs: %s", [j.id for j in scheduler.get_jobs()])
    else:
        logger.info("scheduler disabled by configuration")

    yield

    if scheduler is not None:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="Omaha",
    version="0.1.0",
    description="Document intelligence for NFL prediction. Agents produce evidence, never probabilities.",
    lifespan=lifespan,
)

app.include_router(search_router)


@app.get("/health")
def health(session: DbSession) -> dict[str, Any]:
    now = dt.datetime.now(dt.UTC)
    sources = session.scalars(select(Source).where(Source.enabled.is_(True))).all()

    report: list[dict[str, Any]] = []
    all_ok = True

    for src in sources:
        stale = src.is_stale
        if stale:
            all_ok = False
        age = (now - src.last_success_at).total_seconds() if src.last_success_at else None
        report.append(
            {
                "name": src.name,
                "kind": src.kind,
                "ok": not stale,
                "last_success_at": src.last_success_at.isoformat() if src.last_success_at else None,
                "age_seconds": round(age) if age is not None else None,
                "cadence_seconds": src.cadence_seconds,
                "consecutive_failures": src.consecutive_failures,
                "last_error": src.last_error,
            }
        )

    last_run = session.scalars(select(JobRun).order_by(desc(JobRun.started_at)).limit(1)).first()
    if last_run is not None and not last_run.ok:
        all_ok = False

    return {
        "ok": all_ok,
        "env": settings.env,
        "checked_at": now.isoformat(),
        "scheduler_enabled": settings.scheduler_enabled,
        "last_job": {
            "name": last_run.job_name,
            "started_at": last_run.started_at.isoformat(),
            "ok": last_run.ok,
            "attempted": last_run.sources_attempted,
            "failed": last_run.sources_failed,
            "created": last_run.documents_created,
            "duration_seconds": last_run.duration_seconds,
        }
        if last_run
        else None,
        "sources": report,
    }


@app.get("/stats")
def stats(session: DbSession) -> dict[str, Any]:
    """Corpus size, so you can watch ingestion actually accumulate."""
    doc_count = session.scalar(select(func.count()).select_from(Document))
    src_count = session.scalar(select(func.count()).select_from(Source))
    latest = session.scalar(select(func.max(Document.knowledge_time)))
    return {
        "documents": doc_count or 0,
        "sources": src_count or 0,
        "latest_knowledge_time": latest.isoformat() if latest else None,
    }


@app.get("/jobs")
def jobs(session: DbSession, limit: int = 20) -> dict[str, Any]:
    """Recent scheduled runs — the answer to 'did Wednesday's sweep happen?'"""
    runs = session.scalars(
        select(JobRun).order_by(desc(JobRun.started_at)).limit(min(limit, 100))
    ).all()
    return {
        "runs": [
            {
                "id": r.id,
                "job_name": r.job_name,
                "started_at": r.started_at.isoformat(),
                "ok": r.ok,
                "attempted": r.sources_attempted,
                "failed": r.sources_failed,
                "created": r.documents_created,
                "duration_seconds": r.duration_seconds,
                "error": r.error,
            }
            for r in runs
        ]
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "omaha",
        "version": "0.1.0",
        "ui": "/ui",
        "docs": "/docs",
        "search": "/search?q=...",
    }
