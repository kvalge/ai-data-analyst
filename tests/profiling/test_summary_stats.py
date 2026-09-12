# test_summary_stats.py

"""Tests for numeric and categorical EDA summary stats."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.profiling.eda import SUMMARY_STATS_RESULT_KEYS, summary_stats


def test_summary_stats_sample_sales(sample_sales_csv: Path):
    """Sales fixture: revenue min/mean/max and region nunique."""
    frame = pd.read_csv(sample_sales_csv)
    result = summary_stats(frame)

    assert list(frame.columns) == ["date", "region", "revenue"]
    assert set(result) == SUMMARY_STATS_RESULT_KEYS
    assert result["row_count"] == 5
    assert set(result["numeric"]) == {"revenue"}
    assert set(result["categorical"]) == {"date", "region"}
    assert result["numeric"]["revenue"] == {
        "count": 5,
        "mean": 115.0,
        "median": 120.0,
        "min": 80.0,
        "max": 150.0,
    }
    assert result["categorical"]["region"] == {"nunique": 4}
    assert result["categorical"]["date"] == {"nunique": 5}


def test_summary_stats_empty_numeric_column():
    """No values means count 0 and null moments rather than NaN."""
    frame = pd.DataFrame({"revenue": pd.Series(dtype="float64")})
    result = summary_stats(frame)
    assert result["row_count"] == 0
    assert result["numeric"]["revenue"] == {
        "count": 0,
        "mean": None,
        "median": None,
        "min": None,
        "max": None,
    }


def test_summary_stats_bool_is_categorical():
    """Bool is not summarized as numeric."""
    frame = pd.DataFrame({"flag": [True, False, True]})
    result = summary_stats(frame)
    assert result["numeric"] == {}
    assert result["categorical"]["flag"] == {"nunique": 2}


def test_summary_stats_is_json_serializable(sample_sales_csv: Path):
    """Summary dict must cache as JSON (no numpy leftovers)."""
    result = summary_stats(pd.read_csv(sample_sales_csv))
    encoded = json.dumps(result)
    assert json.loads(encoded) == result
