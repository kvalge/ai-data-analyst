# test_sql_check.py

"""Tests for the generated-SQL deny-list. The checker does not run SQL."""

from __future__ import annotations

import pytest

from src.execution.sql_check import SqlCheckError, check_sql


def test_select_is_allowed():
    """A single SELECT is not on the write/DDL deny-list."""
    check_sql("SELECT date, region, revenue FROM sales")
    check_sql(
        "WITH s AS (SELECT revenue FROM sales) SELECT SUM(revenue) FROM s"
    )
    check_sql("SELECT date, region, revenue FROM sales WHERE region = $1")


def test_empty_sql_is_rejected():
    """Whitespace or comments-only input is empty SQL."""
    with pytest.raises(SqlCheckError, match="empty"):
        check_sql("")
    with pytest.raises(SqlCheckError, match="empty"):
        check_sql("   ")
    with pytest.raises(SqlCheckError, match="empty"):
        check_sql("-- no statement\n")


def test_multi_statement_is_rejected():
    """A second statement is rejected even when both are SELECTs."""
    with pytest.raises(SqlCheckError, match="Multiple"):
        check_sql("SELECT 1; SELECT 2")


def test_non_select_is_rejected():
    """DO, CALL, and other verbs are not SELECT/WITH even if they are not on the deny-list."""
    with pytest.raises(SqlCheckError, match="SELECT or WITH"):
        check_sql("DO $$ BEGIN NULL; END $$")
    with pytest.raises(SqlCheckError, match="SELECT or WITH"):
        check_sql("CALL refresh_sales()")


def test_insert_is_rejected():
    """INSERT is write/DDL and is rejected before the query runs."""
    with pytest.raises(SqlCheckError, match="INSERT"):
        check_sql("INSERT INTO sales (date, region, revenue) VALUES ('2024-01-01', 'North', 10)")


def test_select_into_is_rejected():
    """SELECT INTO is a write even though the statement starts with SELECT."""
    with pytest.raises(SqlCheckError, match="INTO"):
        check_sql("SELECT date, region, revenue INTO tmp FROM sales")


def test_denied_keyword_in_string_is_allowed():
    """A deny-list word inside a string literal is not a write."""
    check_sql("SELECT 'INSERT'")


def test_unclosed_quote_is_rejected():
    """An unclosed quote is a parse reject, not a guessed statement."""
    with pytest.raises(SqlCheckError, match="parsed"):
        check_sql("SELECT 'oops")
