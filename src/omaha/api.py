"""FastAPI application.

/health reports per-source staleness rather than just process liveness. It returns 200
while the process is alive — so the platform doesn't restart a healthy container over
stale upstream data — and sets ok=false when a source is overdue. Point an external
pinger at it and alert on `ok`, not on status code.

(Pattern lifted from the-algo, which learned it the hard way.)
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any

from fastapi import Depends, FastAPI
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from omaha.config import get_settings
from omaha.db.models import Document, Source
from omaha.db.session import get_session

# Annotated dependency alias — keeps Depends() out of argument defaults (ruff B008)
DbSession = Annotated[Session, Depends(get_session)]

settings = get_settings()

app = FastAPI(
    title="Omaha",
    version="0.1.0",
    description="Document intelligence for NFL prediction. Agents produce evidence, never probabilities.",
)


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
        age = (
            (now - src.last_success_at).total_seconds() if src.last_success_at is not None else None
        )
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

    return {
        "ok": all_ok,
        "env": settings.env,
        "checked_at": now.isoformat(),
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


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "omaha", "version": "0.1.0"}
