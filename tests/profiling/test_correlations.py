# test_correlations.py

"""Tests for numeric-only Pearson correlations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.profiling.eda import CORRELATIONS_RESULT_KEYS, correlations


def test_correlations_skips_sales_fixture(sample_sales_csv: Path):
    """Sales has one numeric column, so Pearson is skipped."""
    frame = pd.read_csv(sample_sales_csv)
    result = correlations(frame)

    assert set(result) == CORRELATIONS_RESULT_KEYS
    assert result["row_count"] == 5
    assert result["columns"] == ["revenue"]
    assert result["skipped"] is True
    assert result["pearson"] is None


def test_correlations_perfect_linear_pair():
    """Two numeric columns in a line have Pearson 1.0."""
    frame = pd.DataFrame({"revenue": [1, 2, 3, 4], "quantity": [2, 4, 6, 8]})
    result = correlations(frame)

    assert result["skipped"] is False
    assert result["columns"] == ["revenue", "quantity"]
    pearson = result["pearson"]
    assert pearson is not None
    assert pearson["revenue"]["revenue"] == 1.0
    assert pearson["quantity"]["quantity"] == 1.0
    assert pearson["revenue"]["quantity"] == 1.0
    assert pearson["quantity"]["revenue"] == 1.0


def test_correlations_constant_column_is_null():
    """A constant series has undefined Pearson with the other column."""
    frame = pd.DataFrame({"revenue": [120, 120, 120], "quantity": [1, 2, 3]})
    result = correlations(frame)
    pearson = result["pearson"]
    assert pearson is not None
    assert pearson["revenue"]["quantity"] is None
    assert pearson["quantity"]["revenue"] is None
    assert pearson["quantity"]["quantity"] == 1.0


def test_correlations_bool_is_not_numeric():
    """Bool does not count toward the two-column minimum."""
    frame = pd.DataFrame({"revenue": [1, 2, 3], "flag": [True, False, True]})
    result = correlations(frame)
    assert result["skipped"] is True
    assert result["columns"] == ["revenue"]


def test_correlations_is_json_serializable():
    """Correlation dict must cache as JSON (no numpy leftovers or NaN)."""
    frame = pd.DataFrame({"revenue": [1, 2, 3, 4], "quantity": [2, 4, 6, 8]})
    result = correlations(frame)
    encoded = json.dumps(result)
    assert json.loads(encoded) == result
