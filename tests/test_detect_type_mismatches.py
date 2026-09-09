# test_detect_type_mismatches.py

"""Tests for numeric/date-as-string and mixed-type DQ checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.profiling.dq import TYPE_MISMATCHES_RESULT_KEYS, detect_type_mismatches


def test_detect_type_mismatches_messy_fixture(sample_type_mismatches_csv: Path):
    """Sales-shaped CSV: mixed dates, mixed revenue, clean region."""
    frame = pd.read_csv(sample_type_mismatches_csv)
    result = detect_type_mismatches(frame)

    assert list(frame.columns) == ["date", "region", "revenue"]
    assert set(result) == TYPE_MISMATCHES_RESULT_KEYS
    assert result["row_count"] == 4
    assert result["type_mismatches"] == {
        "date": "mixed",
        "region": None,
        "revenue": "mixed",
    }


def test_detect_type_mismatches_sales_dates_are_strings(sample_sales_csv: Path):
    """Default CSV read leaves ISO dates as strings and revenue as int."""
    result = detect_type_mismatches(pd.read_csv(sample_sales_csv))
    assert result["type_mismatches"]["date"] == "date_as_string"
    assert result["type_mismatches"]["region"] is None
    assert result["type_mismatches"]["revenue"] is None


def test_detect_type_mismatches_numeric_strings():
    """A text column of only numbers is numeric stored as string."""
    frame = pd.DataFrame({"revenue": ["120", "95", "80"]})
    result = detect_type_mismatches(frame)
    assert result["type_mismatches"]["revenue"] == "numeric_as_string"


def test_detect_type_mismatches_unparseable_text_is_not_mixed():
    """Pure notes text is not mixed; mixed needs at least one parseable value."""
    frame = pd.DataFrame(
        {"notes": ["great quarter", "n/a", "pending review"]}
    )
    result = detect_type_mismatches(frame)
    assert result["type_mismatches"]["notes"] is None


def test_detect_type_mismatches_empty_frame_has_no_issues():
    """No rows means no mismatch kinds rather than an error."""
    frame = pd.DataFrame(columns=["date", "region", "revenue"])
    result = detect_type_mismatches(frame)
    assert result["row_count"] == 0
    assert result["type_mismatches"] == {
        "date": None,
        "region": None,
        "revenue": None,
    }


def test_detect_type_mismatches_is_json_serializable(
    sample_type_mismatches_csv: Path,
):
    """Type-mismatch dict must cache as JSON (no numpy leftovers)."""
    result = detect_type_mismatches(pd.read_csv(sample_type_mismatches_csv))
    encoded = json.dumps(result)
    assert json.loads(encoded) == result
