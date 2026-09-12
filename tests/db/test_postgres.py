# test_postgres.py

"""Tests for the optional Postgres connection check (no live database)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings, load_settings
from src.db.postgres import (
    DatabaseError,
    check_postgres_connection,
    postgres_configured,
)


def _settings(tmp_path: Path, **db: str) -> Settings:
    environ = {
        "DB_HOST": "db.local",
        "DB_PORT": "5432",
        "DB_NAME": "analytics",
        "DB_USER": "reader",
        "DB_PASSWORD": "super-secret-password",
    }
    environ.update(db)
    return load_settings(
        environ=environ,
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )


def test_postgres_not_configured_without_name_and_user(tmp_path: Path):
    """Host/port defaults do not count as a configured database."""
    settings = load_settings(
        environ={},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )
    assert postgres_configured(settings) is False


def test_postgres_configured_when_name_and_user_set(tmp_path: Path):
    """DB_NAME and DB_USER are enough to treat Postgres as configured."""
    assert postgres_configured(_settings(tmp_path)) is True


def test_check_raises_before_connect_when_unconfigured(tmp_path: Path):
    """Missing DB_NAME/DB_USER fails without opening a socket."""
    settings = load_settings(
        environ={},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )

    def boom(**_kwargs: object) -> None:
        raise AssertionError("connect should not be called")

    with pytest.raises(DatabaseError, match="not configured"):
        check_postgres_connection(settings, connect=boom)


def test_check_succeeds_when_connect_and_select_work(tmp_path: Path):
    """A usable connection runs SELECT 1 and returns."""
    calls: list[str] = []

    class FakeConnection:
        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def execute(self, sql: str) -> None:
            calls.append(sql)

    def fake_connect(**kwargs: object) -> FakeConnection:
        assert kwargs["host"] == "db.local"
        assert kwargs["dbname"] == "analytics"
        assert kwargs["user"] == "reader"
        assert kwargs["password"] == "super-secret-password"
        return FakeConnection()

    check_postgres_connection(_settings(tmp_path), connect=fake_connect)
    assert calls == ["SELECT 1"]


def test_check_wraps_driver_error_without_password(tmp_path: Path):
    """A failed connect is DatabaseError and does not include the password."""

    def boom(**_kwargs: object) -> None:
        raise OSError("password authentication failed: super-secret-password")

    with pytest.raises(DatabaseError, match="Could not connect") as exc_info:
        check_postgres_connection(_settings(tmp_path), connect=boom)

    message = str(exc_info.value)
    assert "super-secret-password" not in message
    assert "reader" in message
    assert "analytics" in message
