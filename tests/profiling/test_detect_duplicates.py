# test_detect_duplicates.py

"""Tests for full-row duplicate counts on a sample frame."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.profiling.dq import DUPLICATES_RESULT_KEYS, detect_duplicates


def test_detect_duplicates_sample_fixture(sample_duplicates_csv: Path):
    """Three identical sales-shaped rows plus one unique row means two extra copies."""
    frame = pd.read_csv(sample_duplicates_csv)
    result = detect_duplicates(frame)

    assert list(frame.columns) == ["date", "region", "revenue"]
    assert set(result) == DUPLICATES_RESULT_KEYS
    assert result["row_count"] == 4
    assert result["duplicate_row_count"] == 2


def test_detect_duplicates_unique_rows_are_zero(sample_sales_csv: Path):
    """A frame with no repeated rows reports zero extras."""
    result = detect_duplicates(pd.read_csv(sample_sales_csv))
    assert result["duplicate_row_count"] == 0
    assert result["row_count"] == 5


def test_detect_duplicates_empty_frame_is_zero():
    """No rows means zero extras rather than an error."""
    frame = pd.DataFrame(columns=["date", "region", "revenue"])
    result = detect_duplicates(frame)
    assert result["row_count"] == 0
    assert result["duplicate_row_count"] == 0


def test_detect_duplicates_is_json_serializable(sample_duplicates_csv: Path):
    """Duplicates dict must cache as JSON (no numpy leftovers)."""
    result = detect_duplicates(pd.read_csv(sample_duplicates_csv))
    encoded = json.dumps(result)
    assert json.loads(encoded) == result
