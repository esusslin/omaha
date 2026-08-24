"""Day 1 smoke tests. No database required."""

from omaha.config import Settings


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.env == "local"
    assert "postgresql" in s.database_url
    assert s.is_production is False


def test_production_flag() -> None:
    s = Settings(_env_file=None, env="production")
    assert s.is_production is True


def test_app_imports() -> None:
    """Catches import-time errors in the app module."""
    from omaha.api import app

    assert app.title == "Omaha"


def test_importing_the_app_makes_collector_logs_visible() -> None:
    """The scheduler's only output is `logger.info`, and for a while nothing printed it.

    Root defaults to WARNING with no handler, and uvicorn configures only its own
    loggers — so in a container the collector ran silently, reporting neither success
    nor failure. This asserts the app configures logging on import, because a collector
    nobody can observe is indistinguishable from one that isn't running.
    """
    import logging

    import omaha.api  # noqa: F401  — importing is the thing under test

    root = logging.getLogger()
    assert root.handlers, "no root handler — collector logs would go nowhere"
    assert root.level <= logging.INFO, f"root level is {logging.getLevelName(root.level)}"
