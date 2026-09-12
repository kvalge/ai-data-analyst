# sql_check.py

"""Static checks for generated SQL. Parse only; never run the statement."""

from __future__ import annotations

import logging
import re

_LOG = logging.getLogger(__name__)

_DENIED = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "GRANT",
        "COPY",
        "INTO",
    }
)
_DENIED_WORD = re.compile(
    r"\b(" + "|".join(sorted(_DENIED, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)
_ALLOWED_START = re.compile(r"^(SELECT|WITH)\b", flags=re.IGNORECASE)
_DOLLAR_START = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")


class SqlCheckError(ValueError):
    """Generated SQL is empty, multi-statement, not SELECT/WITH, or write/DDL."""


def check_sql(sql: str) -> None:
    """Reject empty, multi-statement, non-SELECT/WITH, or write/DDL SQL.

    The masked statement must start with SELECT or WITH. The deny-list still
    catches writes inside those forms (`SELECT INTO`, `WITH … INSERT`).
    Does not execute the statement. Call this before `query_database` (4.5).
    `$1`-style parameters are left in place. Strings and comments are not
    scanned for deny-list words.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise SqlCheckError("SQL is empty.")
    try:
        masked = _mask_literals_and_comments(sql)
    except SqlCheckError:
        _LOG.info("sql check parse failed")
        raise
    statements = [part.strip() for part in masked.split(";") if part.strip()]
    if not statements:
        raise SqlCheckError("SQL is empty.")
    if len(statements) > 1:
        raise SqlCheckError("Multiple SQL statements are not allowed.")
    statement = statements[0]
    match = _DENIED_WORD.search(statement)
    if match:
        raise SqlCheckError(
            f"SQL write or DDL is not allowed ({match.group(1).upper()})."
        )
    if not _ALLOWED_START.match(statement):
        raise SqlCheckError("SQL must start with SELECT or WITH.")


def _mask_literals_and_comments(sql: str) -> str:
    """Replace strings and comments with spaces so keywords are not spoofed."""
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        char = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if char == "-" and nxt == "-":
            i = _skip_line_comment(sql, i)
            out.append(" ")
            continue
        if char == "/" and nxt == "*":
            i = _skip_block_comment(sql, i)
            out.append(" ")
            continue
        if char == "'":
            i = _skip_quoted(sql, i, quote="'")
            out.append(" ")
            continue
        if char == '"':
            i = _skip_quoted(sql, i, quote='"')
            out.append(" ")
            continue
        dollar = _DOLLAR_START.match(sql, i)
        if dollar is not None:
            i = _skip_dollar_quoted(sql, dollar)
            out.append(" ")
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _skip_line_comment(sql: str, start: int) -> int:
    """Advance past `--` to the end of the line."""
    newline = sql.find("\n", start)
    if newline == -1:
        return len(sql)
    return newline + 1


def _skip_block_comment(sql: str, start: int) -> int:
    """Advance past a `/* */` comment. Postgres comments nest."""
    i = start + 2
    depth = 1
    n = len(sql)
    while i < n and depth:
        if sql.startswith("/*", i):
            depth += 1
            i += 2
            continue
        if sql.startswith("*/", i):
            depth -= 1
            i += 2
            continue
        i += 1
    if depth:
        raise SqlCheckError("SQL could not be parsed.")
    return i


def _skip_quoted(sql: str, start: int, *, quote: str) -> int:
    """Advance past a quoted literal. Doubled quotes are escapes."""
    i = start + 1
    n = len(sql)
    while i < n:
        if sql[i] == quote:
            if i + 1 < n and sql[i + 1] == quote:
                i += 2
                continue
            return i + 1
        i += 1
    raise SqlCheckError("SQL could not be parsed.")


def _skip_dollar_quoted(sql: str, match: re.Match[str]) -> int:
    """Advance past `$tag$...$tag$`. `$1` is a parameter, not a quote."""
    delimiter = match.group(0)
    start = match.end()
    end = sql.find(delimiter, start)
    if end == -1:
        raise SqlCheckError("SQL could not be parsed.")
    return end + len(delimiter)
