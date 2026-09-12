# test_detect_inconsistent_formatting.py

"""Tests for mixed date formats, thousands separators, and whitespace."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.profiling.dq import (
    FORMATTING_RESULT_KEYS,
    detect_inconsistent_formatting,
)


def test_detect_inconsistent_formatting_messy_fixture(
    sample_inconsistent_formatting_csv: Path,
):
    """Sales-shaped CSV: ISO vs slash dates, comma thousands, padded region."""
    frame = pd.read_csv(sample_inconsistent_formatting_csv)
    result = detect_inconsistent_formatting(frame)

    assert list(frame.columns) == ["date", "region", "revenue"]
    assert set(result) == FORMATTING_RESULT_KEYS
    assert result["row_count"] == 3
    assert result["formatting_issues"] == {
        "date": ["mixed_date_formats"],
        "region": ["whitespace"],
        "revenue": ["thousands_separators"],
    }


def test_detect_inconsistent_formatting_sales_is_clean(sample_sales_csv: Path):
    """Uniform ISO dates and typed revenue are not formatting issues."""
    result = detect_inconsistent_formatting(pd.read_csv(sample_sales_csv))
    assert result["formatting_issues"] == {
        "date": [],
        "region": [],
        "revenue": [],
    }


def test_detect_inconsistent_formatting_notes_are_clean():
    """Unparseable notes are not mixed dates, thousands, or whitespace."""
    frame = pd.DataFrame(
        {"notes": ["great quarter", "n/a", "pending review"]}
    )
    result = detect_inconsistent_formatting(frame)
    assert result["formatting_issues"]["notes"] == []


def test_detect_inconsistent_formatting_kind_order():
    """One column with several issues lists kinds in documented order."""
    frame = pd.DataFrame({"messy": [" 2024-01-01", "1/2/24", "1,200"]})
    result = detect_inconsistent_formatting(frame)
    assert result["formatting_issues"]["messy"] == [
        "mixed_date_formats",
        "thousands_separators",
        "whitespace",
    ]


def test_detect_inconsistent_formatting_empty_frame_has_no_issues():
    """No rows means empty issue lists rather than an error."""
    frame = pd.DataFrame(columns=["date", "region", "revenue"])
    result = detect_inconsistent_formatting(frame)
    assert result["row_count"] == 0
    assert result["formatting_issues"] == {
        "date": [],
        "region": [],
        "revenue": [],
    }


def test_detect_inconsistent_formatting_is_json_serializable(
    sample_inconsistent_formatting_csv: Path,
):
    """Formatting dict must cache as JSON (no numpy leftovers)."""
    result = detect_inconsistent_formatting(
        pd.read_csv(sample_inconsistent_formatting_csv)
    )
    encoded = json.dumps(result)
    assert json.loads(encoded) == result
