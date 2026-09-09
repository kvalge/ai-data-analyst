# test_detect_outliers.py

"""Tests for Tukey IQR outlier bounds and counts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.profiling.dq import OUTLIERS_RESULT_KEYS, detect_outliers


def test_detect_outliers_sample_fixture(sample_outliers_csv: Path):
    """Sales-shaped CSV: 10000 is outside the revenue IQR fences."""
    frame = pd.read_csv(sample_outliers_csv)
    result = detect_outliers(frame)

    assert list(frame.columns) == ["date", "region", "revenue"]
    assert set(result) == OUTLIERS_RESULT_KEYS
    assert result["row_count"] == 6
    assert result["outliers"]["date"] is None
    assert result["outliers"]["region"] is None
    report = result["outliers"]["revenue"]
    assert report is not None
    assert report["count"] == 1
    assert report["lower_bound"] < 80
    assert report["upper_bound"] < 10000
    assert report["upper_bound"] > 150


def test_detect_outliers_sales_has_none(sample_sales_csv: Path):
    """The compact sales fixture has no revenue outliers."""
    result = detect_outliers(pd.read_csv(sample_sales_csv))
    report = result["outliers"]["revenue"]
    assert report is not None
    assert report["count"] == 0


def test_detect_outliers_empty_numeric_column():
    """No values means count 0 and null bounds rather than an error."""
    frame = pd.DataFrame({"revenue": pd.Series(dtype="float64")})
    result = detect_outliers(frame)
    assert result["row_count"] == 0
    assert result["outliers"]["revenue"] == {
        "count": 0,
        "lower_bound": None,
        "upper_bound": None,
    }


def test_detect_outliers_constant_column_is_zero():
    """IQR 0 means no value sits outside the fences."""
    frame = pd.DataFrame({"revenue": [120, 120, 120]})
    result = detect_outliers(frame)
    report = result["outliers"]["revenue"]
    assert report is not None
    assert report["count"] == 0
    assert report["lower_bound"] == 120.0
    assert report["upper_bound"] == 120.0


def test_detect_outliers_empty_frame_non_numeric_columns():
    """Empty sales-shaped columns are not numeric, so reports are None."""
    frame = pd.DataFrame(columns=["date", "region", "revenue"])
    result = detect_outliers(frame)
    assert result["row_count"] == 0
    assert result["outliers"] == {
        "date": None,
        "region": None,
        "revenue": None,
    }


def test_detect_outliers_is_json_serializable(sample_outliers_csv: Path):
    """Outliers dict must cache as JSON (no numpy leftovers)."""
    result = detect_outliers(pd.read_csv(sample_outliers_csv))
    encoded = json.dumps(result)
    assert json.loads(encoded) == result
