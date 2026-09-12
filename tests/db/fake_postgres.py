# fake_postgres.py

"""In-memory Postgres connect/cursor for tests. Never opens a socket."""

from __future__ import annotations


class FakeColumn:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeCursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.description = (
            FakeColumn("date"),
            FakeColumn("region"),
            FakeColumn("revenue"),
        )
        self._rows = rows
        self.sql: str | None = None
        self.params: object = None

    def fetchmany(self, size: int) -> list[tuple[object, ...]]:
        return self._rows[:size]


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute(self, sql: str, params: object = ()) -> FakeCursor:
        self._cursor.sql = sql
        self._cursor.params = params
        return self._cursor
