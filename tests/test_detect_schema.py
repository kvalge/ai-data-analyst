# test_detect_schema.py

"""Tests for sample-based schema detection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.profiling.schema import SCHEMA_RESULT_KEYS, detect_schema


def test_detect_schema_sample_sales_csv(sample_sales_csv: Path):
    """The sales fixture has three columns, no sample nulls, and a cheap row count."""
    frame = pd.read_csv(sample_sales_csv)
    result = detect_schema(frame, path=sample_sales_csv)

    assert set(result) == SCHEMA_RESULT_KEYS
    assert result["columns"] == ["date", "region", "revenue"]
    assert result["dtypes"]["region"] in {"object", "str", "string"}
    assert result["dtypes"]["revenue"].startswith("int")
    assert result["null_counts"] == {"date": 0, "region": 0, "revenue": 0}
    assert result["sample_row_count"] == 5
    assert result["file_row_count"] == 5


def test_detect_schema_nulls_and_row_count_from_sample_only(
    tmp_path: Path, sample_sales_csv: Path
):
    """Null counts follow the sample; file_row_count still counts the whole CSV."""
    frame = pd.read_csv(sample_sales_csv, nrows=2)
    result = detect_schema(frame, path=sample_sales_csv)

    assert result["sample_row_count"] == 2
    assert result["file_row_count"] == 5
    assert result["null_counts"]["revenue"] == 0

    sparse = tmp_path / "sparse.csv"
    sparse.write_text("region,revenue\nNorth,\nSouth,10\n", encoding="utf-8")
    sparse_frame = pd.read_csv(sparse)
    sparse_result = detect_schema(sparse_frame, path=sparse)
    assert sparse_result["null_counts"]["revenue"] == 1
    assert sparse_result["file_row_count"] == 2


def test_detect_schema_non_csv_file_row_count_unknown(tmp_path: Path):
    """Excel/JSON have no cheap line count, so file_row_count is None."""
    path = tmp_path / "rows.json"
    path.write_text('[{"a": 1}, {"a": 2}]\n', encoding="utf-8")
    frame = pd.read_json(path)
    result = detect_schema(frame, path=path)

    assert result["columns"] == ["a"]
    assert result["sample_row_count"] == 2
    assert result["file_row_count"] is None


def test_detect_schema_without_path_has_unknown_file_row_count():
    """No path means no filesystem read and file_row_count is unknown."""
    frame = pd.DataFrame({"region": ["North"], "revenue": [120]})
    result = detect_schema(frame)
    assert result["file_row_count"] is None
    assert result["sample_row_count"] == 1
    assert result["columns"] == ["region", "revenue"]


def test_detect_schema_is_json_serializable(sample_sales_csv: Path):
    """Schema dict must cache as JSON (no numpy leftovers)."""
    result = detect_schema(pd.read_csv(sample_sales_csv), path=sample_sales_csv)
    encoded = json.dumps(result)
    assert json.loads(encoded) == result
