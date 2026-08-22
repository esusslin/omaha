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


def build_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=EASTERN)

    # Practice reports: Wed/Thu/Fri at 17:00 ET, an hour after the league deadline.
    scheduler.add_job(
        _run,
        CronTrigger(day_of_week="wed,thu,fri", hour=17, minute=0, timezone=EASTERN),
        args=["injury_sweep", "injury_report"],
        id="injury_sweep",
        max_instances=1,
        coalesce=True,  # a missed fire runs once, not N times
        misfire_grace_time=3600,
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
