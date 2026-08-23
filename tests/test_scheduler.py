"""Scheduler wiring.

These assert on job *configuration* rather than behaviour, because the failures that
have actually happened here were configuration ones: a job filtered on a `kind` no
source had, and a deadline job gated behind a cadence that made it a no-op. Both ran
happily and reported success.
"""

from __future__ import annotations

from omaha.scheduler import EASTERN, build_scheduler


def _jobs():
    return {job.id: job for job in build_scheduler().get_jobs()}


def test_expected_jobs_exist() -> None:
    assert set(_jobs()) == {"injury_sweep", "hourly_sweep", "index"}


def test_injury_sweep_covers_both_source_kinds() -> None:
    """Index sources discover article URLs; `injury_report` covers any direct source.
    Filtering on one kind alone silently matched nothing once already."""
    kinds = _jobs()["injury_sweep"].args[1]
    assert "injury_index" in kinds
    assert "injury_report" in kinds


def test_injury_sweep_ignores_cadence() -> None:
    """The whole point of a 17:00 ET job is to catch the 4pm filing deadline. Cadence
    gating asks 'has enough time passed?', which makes the deadline job skip everything
    whenever the hourly sweep ran recently — and still report success."""
    assert _jobs()["injury_sweep"].args[2] is False


def test_hourly_sweep_respects_cadence() -> None:
    """The polling loop must stay gated, or a 5-minute tick hits every source every
    time. Default is due_only=True, so it takes two args, not three."""
    args = _jobs()["hourly_sweep"].args
    assert len(args) == 2 or args[2] is True


def test_injury_sweep_fires_after_the_league_deadline() -> None:
    """Reports are due 4pm ET Wed/Thu/Fri. An hour later catches late filers."""
    trigger = _jobs()["injury_sweep"].trigger
    fields = {f.name: str(f) for f in trigger.fields}
    assert fields["hour"] == "17"
    assert set(fields["day_of_week"].split(",")) == {"wed", "thu", "fri"}


def test_schedules_are_eastern_not_utc() -> None:
    """NFL deadlines are Eastern. A UTC schedule drifts an hour across daylight saving
    and lands wrong in November — exactly when reports start mattering."""
    for job in _jobs().values():
        assert str(job.trigger.timezone) == str(EASTERN)


def test_jobs_coalesce_and_do_not_overlap() -> None:
    """A missed fire should run once, not N times, and two copies of a sweep against
    one database means duplicate fetches."""
    for job in _jobs().values():
        assert job.coalesce is True
        assert job.max_instances == 1
