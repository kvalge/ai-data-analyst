# test_detect_nulls.py

"""Tests for per-column null counts and percentages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.profiling.dq import NULLS_RESULT_KEYS, detect_nulls


def test_detect_nulls_sample_fixture(sample_nulls_csv: Path):
    """The nulls fixture has a 50% missing revenue column and no missing region."""
    frame = pd.read_csv(sample_nulls_csv)
    result = detect_nulls(frame)

    assert set(result) == NULLS_RESULT_KEYS
    assert result["row_count"] == 4
    assert result["null_counts"] == {"region": 0, "revenue": 2}
    assert result["null_pcts"] == {"region": 0.0, "revenue": 50.0}


def test_detect_nulls_empty_frame_is_zero_percent():
    """No rows means 0 counts and 0.0% rather than division by zero."""
    frame = pd.DataFrame(columns=["region", "revenue"])
    result = detect_nulls(frame)

    assert result["row_count"] == 0
    assert result["null_counts"] == {"region": 0, "revenue": 0}
    assert result["null_pcts"] == {"region": 0.0, "revenue": 0.0}


def test_detect_nulls_is_json_serializable(sample_nulls_csv: Path):
    """Nulls dict must cache as JSON (no numpy leftovers)."""
    result = detect_nulls(pd.read_csv(sample_nulls_csv))
    encoded = json.dumps(result)
    assert json.loads(encoded) == result
