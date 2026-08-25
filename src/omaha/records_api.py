"""Typed injury records, for machine consumers. The seam between Omaha and the-algo.

**This endpoint exists to answer a question nobody else can: is a player absent from the
report, or is our knowledge absent?**

Those look identical in every other data source, and conflating them is the single most
expensive mistake the consumer can make. `the-algo`'s red-team agent has already been
burned by it — a 41% downgrade rate when weather data was missing, because "no value"
was read as "bad value" rather than "no observation". An injury feed that returns an
empty list without saying why invites exactly that.

So every response carries `knowledge`:

    "complete"  — the sources for this team are fresh. An empty record list means the
                  player is not on the report, which is *information*: he's healthy.
    "partial"   — some sources are overdue. Records may be missing. Treat an absence as
                  unknown, not as a clean bill of health.
    "unknown"   — nothing has ever been collected for this team, or everything is stale.
                  An empty list here says nothing at all.

**`as_of` is the other half.** Filtering on `knowledge_time` — when *we* learned a fact,
not when the club published it — is what makes this usable for backtesting rather than
just for a live dashboard. Ask "who was questionable?" as of Friday evening and you get
Friday's answer, uncontaminated by Sunday's inactives list. Without it, every backtest
that touches injuries is quietly training on the future.

Read-only by construction. It fits the hard rule in `the-algo/AGENT_LAYER.md` — nothing
that writes a pick or moves money goes in the PTC allowlist — because there is no write
surface here to allow.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from omaha.config import get_settings
from omaha.db.models import InjuryRecord, Source
from omaha.db.session import get_session
from omaha.extract.prompt import EXTRACTOR_VERSION
from omaha.search_api import RateLimited

settings = get_settings()

DbSession = Annotated[Session, Depends(get_session)]

router = APIRouter()

# A source overdue by more than this multiple of its cadence is stale. Two cadences
# rather than one: a single missed poll is normal jitter, two is a pattern.
STALENESS_MULTIPLE = 2


def _parse_as_of(raw: str | None) -> dt.datetime | None:
    if raw is None:
        return None
    try:
        when = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, f"as_of is not a valid ISO timestamp: {raw}") from exc
    return when if when.tzinfo else when.replace(tzinfo=dt.UTC)


def _knowledge_state(session: Session, team: str, as_of: dt.datetime | None) -> dict[str, Any]:
    """How much should the caller trust an empty result?

    Deliberately computed from *source* health rather than from record counts. A team
    with no records and healthy sources is a team with nobody injured; a team with no
    records and a dead scraper is a team we know nothing about. Counting records cannot
    tell those apart, which is precisely the failure this endpoint exists to prevent.

    When `as_of` is set the question changes: freshness *now* says nothing about whether
    collection was working at some past instant. Rather than imply a guarantee we can't
    make, historical queries report `as_of_historical` and leave the judgement to the
    caller, who knows what tolerance the backtest needs.
    """
    sources = session.scalars(
        select(Source).where(Source.team == team, Source.enabled.is_(True))
    ).all()

    if not sources:
        return {"knowledge": "unknown", "reason": f"no enabled sources registered for {team}"}

    now = dt.datetime.now(dt.UTC)
    fresh = stale = never = 0
    for source in sources:
        if source.last_success_at is None:
            never += 1
            continue
        age = (now - source.last_success_at).total_seconds()
        if age > source.cadence_seconds * STALENESS_MULTIPLE:
            stale += 1
        else:
            fresh += 1

    detail = {"sources": len(sources), "fresh": fresh, "stale": stale, "never_succeeded": never}

    if as_of is not None:
        return {
            "knowledge": "as_of_historical",
            "reason": (
                "current source health does not describe collection at a past instant; "
                "judge sufficiency against the backtest's own tolerance"
            ),
            **detail,
        }
    if fresh == 0:
        return {"knowledge": "unknown", "reason": "no source for this team is fresh", **detail}
    if stale or never:
        return {
            "knowledge": "partial",
            "reason": f"{stale + never} of {len(sources)} sources overdue",
            **detail,
        }
    return {"knowledge": "complete", "reason": "all sources fresh", **detail}


def _serialise(record: InjuryRecord) -> dict[str, Any]:
    return {
        "player_name": record.player_name,
        "player_id": record.player_id,
        "team": record.team,
        "position": record.position,
        "injury": record.injury,
        "practice_status": record.practice_status,
        "game_status": record.game_status,
        "report_day": record.report_day,
        "evidence": record.evidence,
        "knowledge_time": record.knowledge_time.isoformat(),
        "published_time": record.published_time.isoformat() if record.published_time else None,
        "chunk_id": record.chunk_id,
        "document_id": record.document_id,
    }


@router.get("/injuries")
def injuries(
    session: DbSession,
    team: Annotated[str, Query(min_length=2, max_length=4, description="e.g. PHI")],
    as_of: Annotated[str | None, Query(description="ISO 8601; filters on knowledge_time")] = None,
    since: Annotated[
        str | None, Query(description="ISO 8601; only facts learned after this")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    _limited: RateLimited = None,
) -> dict[str, Any]:
    """Typed injury facts for one team, with an explicit statement of what we know.

    One team per call rather than a game id, because Omaha has no concept of a game —
    that mapping lives in `the-algo`, which owns the schedule. Pushing it here would
    duplicate a source of truth and couple two systems that are usefully independent.
    """
    abbr = team.strip().upper()
    when = _parse_as_of(as_of)
    after = _parse_as_of(since)

    statement = select(InjuryRecord).where(
        InjuryRecord.team == abbr,
        InjuryRecord.extractor_version == EXTRACTOR_VERSION,
    )
    if when is not None:
        statement = statement.where(InjuryRecord.knowledge_time <= when)
    if after is not None:
        statement = statement.where(InjuryRecord.knowledge_time > after)

    records = session.scalars(
        statement.order_by(InjuryRecord.knowledge_time.desc()).limit(limit)
    ).all()

    return {
        "team": abbr,
        "as_of": when.isoformat() if when else None,
        "since": after.isoformat() if after else None,
        "extractor_version": EXTRACTOR_VERSION,
        **_knowledge_state(session, abbr, when),
        "count": len(records),
        "records": [_serialise(r) for r in records],
    }


@router.get("/injuries/trajectory")
def trajectory(
    session: DbSession,
    team: Annotated[str, Query(min_length=2, max_length=4)],
    player: Annotated[str, Query(min_length=2, description="substring match on name")],
    as_of: Annotated[str | None, Query(description="ISO 8601")] = None,
    _limited: RateLimited = None,
) -> dict[str, Any]:
    """One player's practice sequence — the shape the model wants, not a flat status.

    DNP → LIMITED → FULL and DNP → DNP → LIMITED end at different places and mean
    different things, and a single final status flattens both to "limited". The measured
    result in `research/practice_signal.py` (+0.054 AUC on the Questionable panel) is
    what makes this worth exposing separately.

    **Scoped per report.** `latest` is the most recent report — the one a model asking
    "is he playing Sunday?" needs. `history` holds earlier reports for anyone modelling
    recovery over time. Flattening the two together was the first version of this
    endpoint and it produced a sequence spanning four unrelated injuries.

    Records whose day the club never published are still returned, with
    `report_day: null` — dropping them would hide about a third of what we have, and the
    caller can decide whether an unsequenced fact is useful.
    """
    abbr = team.strip().upper()
    when = _parse_as_of(as_of)

    statement = select(InjuryRecord).where(
        InjuryRecord.team == abbr,
        InjuryRecord.player_name.ilike(f"%{player.strip()}%"),
        InjuryRecord.extractor_version == EXTRACTOR_VERSION,
    )
    if when is not None:
        statement = statement.where(InjuryRecord.knowledge_time <= when)

    records = session.scalars(statement.order_by(InjuryRecord.knowledge_time)).all()

    # **Group by source document, because a trajectory is a week — not a career.**
    #
    # The first version concatenated every record a player had, which for Jalen Carter
    # produced a 24-element list spanning Hip, Shoulders, Heel and Illness across a whole
    # season. DNP -> LIMITED -> FULL only means anything within one report week; strung
    # across weeks it is noise wearing the shape of a signal, and a consumer would have
    # had no way to tell.
    #
    # One article covers one report period, so `document_id` is the natural boundary.
    # Season and week would be cleaner but are frequently null on these documents —
    # grouping on a field that's often absent would silently merge everything into one
    # bucket, which is the bug this fix exists to remove.
    # Grouped by source document. **A document is not always a week.** Some clubs publish
    # one article covering Wed/Thu/Fri; others publish daily, in which case each group is a
    # single day. The field is named `reports` rather than `weeks` because calling a daily
    # article a week is a claim the data doesn't support — Atlanta returns 22 of them for
    # one player across two months.
    episodes: dict[int, list[InjuryRecord]] = {}
    for record in records:
        episodes.setdefault(record.document_id, []).append(record)

    def as_report(rows: list[InjuryRecord]) -> dict[str, Any]:
        # Within a week, order by the day the practice happened, not by when we learned
        # it — a single fetch stamps every row in an article with the same knowledge_time,
        # so knowledge order says nothing about Wednesday before Thursday.
        order = {day: i for i, day in enumerate(("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"))}
        ordered = sorted(rows, key=lambda r: order.get(r.report_day or "", 99))
        return {
            "document_id": rows[0].document_id,
            "published_time": rows[0].published_time.isoformat()
            if rows[0].published_time
            else None,
            "knowledge_time": min(r.knowledge_time for r in rows).isoformat(),
            "injury": next((r.injury for r in ordered if r.injury), None),
            "practice_sequence": [r.practice_status for r in ordered if r.practice_status],
            "game_status": next((r.game_status for r in reversed(ordered) if r.game_status), None),
            "days": [
                {
                    "report_day": r.report_day,
                    "practice_status": r.practice_status,
                    "game_status": r.game_status,
                }
                for r in ordered
            ],
        }

    reports = [as_report(rows) for rows in episodes.values()]
    reports.sort(key=lambda r: r["knowledge_time"])
    latest = reports[-1] if reports else None

    return {
        "team": abbr,
        "player": player,
        "as_of": when.isoformat() if when else None,
        **_knowledge_state(session, abbr, when),
        "records": len(records),
        "reports": len(reports),
        # The current week's trajectory — what a model asking "is he playing Sunday?"
        # wants. Prior reports are available below for anyone modelling recovery.
        "latest": latest,
        "history": reports[:-1] if len(reports) > 1 else [],
    }


@router.get("/injuries/summary")
def summary(session: DbSession, _limited: RateLimited = None) -> dict[str, Any]:
    """What's in the store, by team. Cheap enough to poll, useful for spotting a club
    whose collection has quietly stopped producing records."""
    rows = session.execute(
        select(
            InjuryRecord.team,
            func.count(InjuryRecord.id),
            func.max(InjuryRecord.knowledge_time),
        )
        .where(InjuryRecord.extractor_version == EXTRACTOR_VERSION)
        .group_by(InjuryRecord.team)
        .order_by(func.count(InjuryRecord.id).desc())
    ).all()

    return {
        "extractor_version": EXTRACTOR_VERSION,
        "teams": [
            {"team": team, "records": count, "latest": latest.isoformat() if latest else None}
            for team, count, latest in rows
        ],
        "total": sum(count for _, count, _ in rows),
    }
