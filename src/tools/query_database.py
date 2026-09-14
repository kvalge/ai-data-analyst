# query_database.py

"""Tool: run one checked, parameterized SELECT/WITH against Postgres."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any, cast

import psycopg

from src.config import Settings
from src.db.postgres import DatabaseError, postgres_configured
from src.execution.sql_check import check_sql
from src.storage.sources import postgres_source_id
from src.tools.contracts import ToolContract

_LOG = logging.getLogger(__name__)

QUERY_DATABASE = ToolContract(
    name="query_database",
    description=(
        "Run one parameterized SELECT or WITH against the enabled Postgres "
        "source. Provide connection_id (the postgres source_id) and sql. "
        "Optional params bind to $1, $2, …. Never returns the password or "
        "an unbounded result set."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "connection_id": {"type": "string"},
            "sql": {"type": "string"},
            "params": {"type": "array"},
        },
        "required": ["connection_id", "sql"],
        "additionalProperties": False,
    },
    result_schema={
        "type": "object",
        "properties": {
            "connection_id": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "row_count": {"type": "integer"},
            "truncated": {"type": "boolean"},
            "rows": {"type": "array", "items": {"type": "object"}},
        },
        "required": [
            "connection_id",
            "columns",
            "row_count",
            "truncated",
            "rows",
        ],
        "additionalProperties": False,
    },
)


def query_database(
    *,
    settings: Settings,
    sql: str,
    connection_id: str,
    n_rows: int,
    include_postgres: bool = False,
    params: Sequence[Any] | None = None,
    connect: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Check SQL, then run it with bound parameters. Never the full table.

    `settings`, `include_postgres`, `n_rows`, and `connect` are injected by
    the app or tests, not the LLM. Default pytest uses a fake `connect`.
    """
    if not include_postgres or not postgres_configured(settings):
        raise DatabaseError("Postgres is not configured or not enabled.")
    expected = postgres_source_id(settings)
    if connection_id != expected:
        raise DatabaseError("Unknown connection_id.")
    check_sql(sql)
    if n_rows < 1:
        raise DatabaseError("n_rows must be at least 1.")
    bound = () if params is None else tuple(params)
    opener = connect if connect is not None else psycopg.connect
    _LOG.info("query_database connection_id=%s n_rows=%s", connection_id, n_rows)
    try:
        with opener(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            connect_timeout=settings.db_connect_timeout_s,
        ) as connection:
            cursor = connection.execute(cast(Any, sql), bound)
            columns = _column_names(cursor)
            fetched = list(cursor.fetchmany(n_rows + 1))
    except (OSError, psycopg.Error) as exc:
        # Driver errors often embed SQL and bound values. Do not log exc.
        _LOG.error("query_database failed: %s", type(exc).__name__)
        raise DatabaseError(
            f"Could not query Postgres at {settings.db_host}:"
            f"{settings.db_port}/{settings.db_name} as {settings.db_user}."
        ) from exc
    truncated = len(fetched) > n_rows
    records = fetched[:n_rows]
    rows = [_row_to_object(columns, row) for row in records]
    return {
        "connection_id": connection_id,
        "columns": columns,
        "row_count": len(rows),
        "truncated": truncated,
        "rows": rows,
    }


def _column_names(cursor: Any) -> list[str]:
    """Read cursor.description names. Empty when the cursor has no fields."""
    description = getattr(cursor, "description", None)
    if not description:
        return []
    names: list[str] = []
    for item in description:
        name = getattr(item, "name", None)
        names.append(str(name if name is not None else item[0]))
    return names


def _row_to_object(columns: list[str], row: Sequence[Any]) -> dict[str, Any]:
    """Map one driver row to a JSON-safe dict. Not the raw tuple."""
    return {
        columns[index]: _json_safe_cell(value)
        for index, value in enumerate(row)
        if index < len(columns)
    }


def _json_safe_cell(value: Any) -> Any:
    """Convert a driver value so it can sit in a tool result."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return str(value)
