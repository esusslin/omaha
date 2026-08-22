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
