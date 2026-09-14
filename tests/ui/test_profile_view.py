# test_profile_view.py

"""Tests for profile UI table builders from a profile_source result.

Sales-profile tests are one builder each so a failure names the helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.config import DEFAULT_MAX_UPLOAD_BYTES, DEFAULT_SAMPLE_N_ROWS
from src.storage.registry import save_file_source
from src.tools.profile_source import profile_source
from src.ui.profile import (
    categorical_summary_rows,
    categorical_top_rows,
    formatting_rows,
    histogram_rows,
    mismatch_rows,
    null_rows,
    numeric_summary_rows,
    outlier_rows,
    pearson_rows,
    schema_rows,
)

_MAX = DEFAULT_MAX_UPLOAD_BYTES


def _register_sales(tmp_path: Path, sample_sales_csv: Path):
    upload_dir = tmp_path / "uploads"
    cache_dir = tmp_path / "cache"
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(incoming, upload_dir, original_name="sales.csv")
    return upload_dir, cache_dir, saved


@pytest.fixture
def sales_profile(tmp_path: Path, sample_sales_csv: Path) -> dict[str, Any]:
    """One profile_source result shared by the per-builder sales tests."""
    upload_dir, cache_dir, saved = _register_sales(tmp_path, sample_sales_csv)
    return profile_source(
        upload_dir=upload_dir,
        cache_dir=cache_dir,
        n_rows=DEFAULT_SAMPLE_N_ROWS,
        max_bytes=_MAX,
        source_id=saved.source_id,
    )


def test_schema_rows_from_sales_profile(sales_profile: dict[str, Any]):
    """Schema table is one row per column."""
    assert [row["column"] for row in schema_rows(sales_profile["schema"])] == [
        "date",
        "region",
        "revenue",
    ]


def test_null_rows_from_sales_profile(sales_profile: dict[str, Any]):
    """Nulls table is one row per column."""
    assert [row["column"] for row in null_rows(sales_profile["dq"]["nulls"])] == [
        "date",
        "region",
        "revenue",
    ]


def test_mismatch_rows_from_sales_profile(sales_profile: dict[str, Any]):
    """Sales date stored as text is the only type-mismatch row."""
    assert mismatch_rows(
        sales_profile["dq"]["type_mismatches"]["type_mismatches"]
    ) == [{"column": "date", "kind": "date_as_string"}]


def test_formatting_rows_from_sales_profile(sales_profile: dict[str, Any]):
    """Sales fixture has no formatting-issue rows."""
    assert formatting_rows(sales_profile["dq"]["formatting"]["formatting_issues"]) == []


def test_outlier_rows_from_sales_profile(sales_profile: dict[str, Any]):
    """Only numeric revenue appears in the outlier table."""
    outliers = outlier_rows(sales_profile["dq"]["outliers"]["outliers"])
    assert [row["column"] for row in outliers] == ["revenue"]


def test_numeric_summary_rows_from_sales_profile(sales_profile: dict[str, Any]):
    """Numeric summary lists revenue only."""
    rows = numeric_summary_rows(sales_profile["eda"]["summary"]["numeric"])
    assert [row["column"] for row in rows] == ["revenue"]


def test_categorical_summary_rows_from_sales_profile(sales_profile: dict[str, Any]):
    """Categorical summary lists date and region."""
    rows = categorical_summary_rows(sales_profile["eda"]["summary"]["categorical"])
    assert {row["column"] for row in rows} == {"date", "region"}


def test_pearson_rows_from_sales_profile(sales_profile: dict[str, Any]):
    """One numeric column means Pearson is skipped and the table is empty."""
    assert pearson_rows(sales_profile["eda"]["correlations"]) == []


def test_mismatch_rows_drop_clean_columns():
    """Columns with no type-mismatch kind are omitted."""
    assert mismatch_rows({"date": None, "amount": "numeric_as_string"}) == [
        {"column": "amount", "kind": "numeric_as_string"}
    ]


def test_formatting_rows_drop_clean_columns():
    """Columns with an empty kinds list are omitted."""
    assert formatting_rows({"date": [], "note": ["whitespace"]}) == [
        {"column": "note", "kinds": "whitespace"}
    ]


def test_histogram_rows():
    """Numeric bins become low/high/count rows."""
    assert histogram_rows({"counts": [2, 1], "edges": [0.0, 10.0, 20.0]}) == [
        {"bin": 0, "low": 0.0, "high": 10.0, "count": 2},
        {"bin": 1, "low": 10.0, "high": 20.0, "count": 1},
    ]


def test_categorical_top_rows_include_other():
    """A nonzero other_count is kept as a trailing row."""
    assert categorical_top_rows(
        {"top": [{"value": "North", "count": 3}], "other_count": 2}
    ) == [
        {"value": "North", "count": 3},
        {"value": "(other)", "count": 2},
    ]


def test_pearson_rows_square_matrix():
    """A computed Pearson payload becomes one row per numeric column."""
    corr = {
        "columns": ["a", "b"],
        "pearson": {"a": {"a": 1.0, "b": 0.5}, "b": {"a": 0.5, "b": 1.0}},
        "skipped": False,
        "row_count": 3,
    }
    assert pearson_rows(corr) == [
        {"column": "a", "a": 1.0, "b": 0.5},
        {"column": "b", "a": 0.5, "b": 1.0},
    ]


def test_pearson_rows_skipped_is_empty():
    """A skipped Pearson payload has no table rows."""
    assert pearson_rows(
        {"columns": ["a"], "pearson": None, "skipped": True, "row_count": 3}
    ) == []
