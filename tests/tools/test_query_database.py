# test_query_database.py

"""Tests for query_database. Default pytest uses a fake connect, not a live DB."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.agent.audit import audit_log_path, default_audit_dir
from src.agent.execute import run_allowlisted_tool
from src.config import load_settings
from src.db.postgres import DatabaseError
from src.execution.sql_check import SqlCheckError
from src.storage.sources import postgres_source_id
from src.tools.query_database import QUERY_DATABASE, query_database
from tests.tool_schema import assert_keys_match_required

_SELECT = "SELECT date, region, revenue FROM sales WHERE region = $1"
_SALES_ROW = ("2024-01-01", "North", 10.0)


class _Column:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeCursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.description = (_Column("date"), _Column("region"), _Column("revenue"))
        self._rows = rows
        self.sql: str | None = None
        self.params: object = None

    def fetchmany(self, size: int) -> list[tuple[object, ...]]:
        return self._rows[:size]


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute(self, sql: str, params: object = ()) -> _FakeCursor:
        self._cursor.sql = sql
        self._cursor.params = params
        return self._cursor


def _settings(tmp_path: Path):
    return load_settings(
        environ={
            "DB_HOST": "db.local",
            "DB_PORT": "5432",
            "DB_NAME": "analytics",
            "DB_USER": "reader",
            "DB_PASSWORD": "super-secret-password",
        },
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )


def _run(tmp_path: Path, *, sql: str, cursor: _FakeCursor, **kwargs: object):
    settings = _settings(tmp_path)
    connection = _FakeConnection(cursor)

    def fake_connect(**_kw: object) -> _FakeConnection:
        return connection

    return query_database(
        settings=settings,
        sql=sql,
        connection_id=postgres_source_id(settings),
        n_rows=2,
        include_postgres=True,
        connect=fake_connect,
        **kwargs,
    )


def test_select_returns_bounded_rows(tmp_path: Path):
    """A checked SELECT returns columns and a bounded head, not a dump."""
    cursor = _FakeCursor([_SALES_ROW, ("2024-01-02", "South", 5.0)])
    result = _run(tmp_path, sql=_SELECT, cursor=cursor, params=["North"])
    assert result["columns"] == ["date", "region", "revenue"]
    assert result["row_count"] == 2
    assert result["truncated"] is False
    assert result["rows"][0]["region"] == "North"
    assert cursor.sql == _SELECT
    assert cursor.params == ("North",)
    assert_keys_match_required(result, QUERY_DATABASE.result_schema)
    assert "super-secret-password" not in str(result)


def test_extra_row_is_truncated(tmp_path: Path):
    """Fetching past n_rows marks truncated and drops the extra row."""
    cursor = _FakeCursor(
        [
            _SALES_ROW,
            ("2024-01-02", "South", 5.0),
            ("2024-01-03", "East", 7.0),
        ]
    )
    result = _run(tmp_path, sql=_SELECT, cursor=cursor, params=["North"])
    assert result["row_count"] == 2
    assert result["truncated"] is True
    assert len(result["rows"]) == 2


def test_insert_is_rejected_before_connect(tmp_path: Path):
    """Write SQL fails the checker; connect is not opened."""
    settings = _settings(tmp_path)

    def boom(**_kwargs: object) -> None:
        raise AssertionError("connect should not be called")

    with pytest.raises(SqlCheckError, match="INSERT"):
        query_database(
            settings=settings,
            sql="INSERT INTO sales (date) VALUES ('2024-01-01')",
            connection_id=postgres_source_id(settings),
            n_rows=2,
            include_postgres=True,
            connect=boom,
        )


def test_unknown_connection_id_is_rejected_before_connect(tmp_path: Path):
    """A file or invented id is not queried."""
    settings = _settings(tmp_path)

    def boom(**_kwargs: object) -> None:
        raise AssertionError("connect should not be called")

    with pytest.raises(DatabaseError, match="Unknown connection_id"):
        query_database(
            settings=settings,
            sql=_SELECT,
            connection_id="file-abc",
            n_rows=2,
            include_postgres=True,
            connect=boom,
        )


def test_disabled_postgres_is_rejected_before_connect(tmp_path: Path):
    """The sidebar must enable the configured DB before a query runs."""
    settings = _settings(tmp_path)

    def boom(**_kwargs: object) -> None:
        raise AssertionError("connect should not be called")

    with pytest.raises(DatabaseError, match="not enabled"):
        query_database(
            settings=settings,
            sql=_SELECT,
            connection_id=postgres_source_id(settings),
            n_rows=2,
            include_postgres=False,
            connect=boom,
        )


def test_driver_error_omits_password(tmp_path: Path):
    """A failed query is DatabaseError and does not include the password."""
    settings = _settings(tmp_path)

    def boom(**_kwargs: object) -> None:
        raise OSError("password authentication failed: super-secret-password")

    with pytest.raises(DatabaseError, match="Could not query") as exc_info:
        query_database(
            settings=settings,
            sql=_SELECT,
            connection_id=postgres_source_id(settings),
            n_rows=2,
            include_postgres=True,
            connect=boom,
        )
    assert "super-secret-password" not in str(exc_info.value)


def test_driver_error_log_omits_sql_and_params(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A driver failure logs the exception type, not SQL or bound values."""
    settings = _settings(tmp_path)

    def boom(**_kwargs: object) -> None:
        raise OSError(f"syntax error in {_SELECT} params=('North',)")

    with caplog.at_level(logging.ERROR, logger="src.tools.query_database"):
        with pytest.raises(DatabaseError, match="Could not query"):
            query_database(
                settings=settings,
                sql=_SELECT,
                connection_id=postgres_source_id(settings),
                n_rows=2,
                include_postgres=True,
                params=["North"],
                connect=boom,
            )
    assert "query_database failed: OSError" in caplog.text
    assert _SELECT not in caplog.text
    assert "North" not in caplog.text


def test_query_database_audit_line_has_connection_id_not_sql(tmp_path: Path):
    """run_allowlisted_tool injects connect and writes identities only."""
    settings = _settings(tmp_path)
    connection_id = postgres_source_id(settings)
    cursor = _FakeCursor([_SALES_ROW])
    connection = _FakeConnection(cursor)

    def injected_connect(**_kw: object) -> _FakeConnection:
        return connection

    def llm_connect(**_kw: object) -> None:
        raise AssertionError("LLM-supplied connect must not run")

    result = run_allowlisted_tool(
        "query_database",
        {
            "connection_id": connection_id,
            "sql": _SELECT,
            "params": ["North"],
            "connect": llm_connect,
            "settings": object(),
        },
        settings,
        include_postgres=True,
        connect=injected_connect,
    )
    assert result["rows"][0]["region"] == "North"
    assert cursor.params == ("North",)
    line = audit_log_path(default_audit_dir(settings.upload_dir)).read_text(
        encoding="utf-8"
    )
    record = json.loads(line)
    assert record["tool"] == "query_database"
    assert record["source_id"] == connection_id
    assert _SELECT not in line
    assert "North" not in line
    assert "super-secret-password" not in line


def test_query_database_contract_is_complete():
    """The MCP-shaped contract has a name, description, and JSON schemas."""
    assert QUERY_DATABASE.name == "query_database"
    assert QUERY_DATABASE.description
    assert QUERY_DATABASE.input_schema["required"] == ["connection_id", "sql"]
    assert QUERY_DATABASE.result_schema["required"] == [
        "connection_id",
        "columns",
        "row_count",
        "truncated",
        "rows",
    ]
