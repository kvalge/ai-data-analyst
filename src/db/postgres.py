# postgres.py

"""Optional Postgres connection check. The DB role must be read-only."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import psycopg

from src.config import Settings

_LOG = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Postgres is not configured or the connection is not usable."""


def postgres_configured(settings: Settings) -> bool:
    """True when DB_NAME and DB_USER are set. Host/port have defaults."""
    return bool(settings.db_name and settings.db_user)


def check_postgres_connection(
    settings: Settings,
    *,
    connect: Callable[..., Any] | None = None,
) -> None:
    """Open a short-lived connection and run SELECT 1.

    Read-only access is a database-role requirement, not enforced here.
    `connect` is injected in tests; the app uses `psycopg.connect`.
    """
    if not postgres_configured(settings):
        raise DatabaseError("Postgres is not configured. Set DB_NAME and DB_USER.")

    opener = connect if connect is not None else psycopg.connect
    try:
        with opener(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            connect_timeout=settings.db_connect_timeout_s,
        ) as connection:
            connection.execute("SELECT 1")
    except (OSError, psycopg.Error) as exc:
        _LOG.exception("Postgres connection test failed")
        raise DatabaseError(
            f"Could not connect to Postgres at {settings.db_host}:"
            f"{settings.db_port}/{settings.db_name} as {settings.db_user}."
        ) from exc
    _LOG.info(
        "Postgres connection ok host=%s port=%s db=%s user=%s",
        settings.db_host,
        settings.db_port,
        settings.db_name,
        settings.db_user,
    )
