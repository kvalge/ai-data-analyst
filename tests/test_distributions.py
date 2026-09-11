# test_distributions.py

"""Tests for numeric histograms and categorical top-N counts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.profiling.eda import DISTRIBUTIONS_RESULT_KEYS, distributions


def test_distributions_sample_sales(sample_sales_csv: Path):
    """Sales fixture: 10 revenue bins summing to 5 rows; North is top region."""
    frame = pd.read_csv(sample_sales_csv)
    result = distributions(frame)

    assert list(frame.columns) == ["date", "region", "revenue"]
    assert set(result) == DISTRIBUTIONS_RESULT_KEYS
    assert result["row_count"] == 5
    hist = result["numeric"]["revenue"]
    assert hist["counts"] == [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    assert hist["edges"][0] == 80.0
    assert hist["edges"][-1] == 150.0
    assert len(hist["edges"]) == 11
    assert sum(hist["counts"]) == 5
    assert result["categorical"]["region"] == {
        "top": [
            {"value": "North", "count": 2},
            {"value": "South", "count": 1},
            {"value": "East", "count": 1},
            {"value": "West", "count": 1},
        ],
        "other_count": 0,
    }
    assert result["categorical"]["date"]["other_count"] == 0
    assert len(result["categorical"]["date"]["top"]) == 5


def test_distributions_empty_numeric_column():
    """No values means empty bins rather than NaN edges."""
    frame = pd.DataFrame({"revenue": pd.Series(dtype="float64")})
    result = distributions(frame)
    assert result["numeric"]["revenue"] == {"counts": [], "edges": []}


def test_distributions_top_n_overflow():
    """Values past the top-N are counted in other_count."""
    labels = [f"c{index}" for index in range(12)]
    result = distributions(pd.DataFrame({"label": labels}))
    report = result["categorical"]["label"]
    assert len(report["top"]) == 10
    assert report["other_count"] == 2
    assert sum(item["count"] for item in report["top"]) + report["other_count"] == 12


def test_distributions_bool_is_categorical():
    """Bool uses value counts, not a histogram; numpy.bool_ becomes JSON bool."""
    frame = pd.DataFrame(
        {"flag": np.array([True, False, True], dtype=np.bool_)}
    )
    result = distributions(frame)
    assert result["numeric"] == {}
    top = result["categorical"]["flag"]["top"]
    assert top[0]["value"] is True
    assert top[1]["value"] is False
    assert top == [
        {"value": True, "count": 2},
        {"value": False, "count": 1},
    ]


def test_distributions_is_json_serializable(sample_sales_csv: Path):
    """Distributions dict must cache as JSON (no numpy leftovers)."""
    result = distributions(pd.read_csv(sample_sales_csv))
    encoded = json.dumps(result)
    assert json.loads(encoded) == result
