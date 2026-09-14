# test_read_file_sample.py

"""Tests for read_file_sample and its tool contract."""

import json
from pathlib import Path

import pytest

from src.config import DEFAULT_MAX_UPLOAD_BYTES, DEFAULT_SAMPLE_N_ROWS
from src.storage.registry import save_file_source
from src.tools.read_sample import READ_FILE_SAMPLE, read_file_sample
from src.validation.data_files import FileValidationError
from tests.tool_schema import assert_keys_match_required

_MAX = DEFAULT_MAX_UPLOAD_BYTES


def test_read_sample_by_source_id(tmp_path: Path, sample_sales_csv: Path):
    """A registered CSV returns columns, dtypes, and a bounded head."""
    upload_dir = tmp_path / "uploads"
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(incoming, upload_dir, original_name="sales.csv")

    result = read_file_sample(
        upload_dir=upload_dir,
        n_rows=2,
        max_bytes=_MAX,
        source_id=saved.source_id,
    )

    assert result["columns"] == ["date", "region", "revenue"]
    assert "revenue" in result["dtypes"]
    assert result["row_count"] == 2
    assert len(result["rows"]) == 2
    assert result["rows"][0]["region"] == "North"
    assert result["rows"][0]["revenue"] == 120
    assert all(row.get("region") != "West" for row in result["rows"])
    assert_keys_match_required(result, READ_FILE_SAMPLE.result_schema)


def test_read_sample_by_path_under_upload_dir(tmp_path: Path, sample_sales_csv: Path):
    """A stored filename under upload_dir can be sampled without source_id."""
    upload_dir = tmp_path / "uploads"
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(incoming, upload_dir, original_name="sales.csv")

    result = read_file_sample(
        upload_dir=upload_dir,
        n_rows=DEFAULT_SAMPLE_N_ROWS,
        max_bytes=_MAX,
        path=saved.stored_path.name,
    )

    assert result["row_count"] == 5
    assert result["rows"][-1]["region"] == "West"


def test_read_sample_rejects_path_outside_upload_dir(
    tmp_path: Path, sample_sales_csv: Path
):
    """A path outside UPLOAD_DIR is rejected even if the file is valid."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    with pytest.raises(FileValidationError, match="outside"):
        read_file_sample(
            upload_dir=upload_dir,
            n_rows=2,
            max_bytes=_MAX,
            path=sample_sales_csv,
        )


def test_read_sample_unknown_source_id(tmp_path: Path):
    """An unknown source_id is a domain error, not an empty sample."""
    with pytest.raises(FileValidationError, match="Unknown source_id"):
        read_file_sample(
            upload_dir=tmp_path / "uploads",
            n_rows=2,
            max_bytes=_MAX,
            source_id="file-missing",
        )


def test_read_sample_requires_exactly_one_locator(tmp_path: Path):
    """Both missing or both set is invalid."""
    upload_dir = tmp_path / "uploads"
    with pytest.raises(FileValidationError, match="exactly one"):
        read_file_sample(upload_dir=upload_dir, n_rows=2, max_bytes=_MAX)
    with pytest.raises(FileValidationError, match="exactly one"):
        read_file_sample(
            upload_dir=upload_dir,
            n_rows=2,
            max_bytes=_MAX,
            source_id="file-x",
            path="sales.csv",
        )


def test_read_sample_rejects_n_rows_below_one(tmp_path: Path):
    """n_rows must be a positive count."""
    with pytest.raises(FileValidationError, match="n_rows"):
        read_file_sample(
            upload_dir=tmp_path / "uploads",
            n_rows=0,
            max_bytes=_MAX,
            source_id="file-x",
        )


def test_read_sample_rows_are_json_serializable(tmp_path: Path):
    """rows must json.dumps without Timestamp or numpy leftovers."""
    upload_dir = tmp_path / "uploads"
    incoming = tmp_path / "events.json"
    incoming.write_text(
        '[{"ts": "2024-01-01T12:00:00Z", "amount": 1.5}]\n',
        encoding="utf-8",
    )
    saved = save_file_source(incoming, upload_dir, original_name="events.json")

    result = read_file_sample(
        upload_dir=upload_dir,
        n_rows=1,
        max_bytes=_MAX,
        source_id=saved.source_id,
    )

    encoded = json.dumps(result["rows"])
    assert json.loads(encoded) == result["rows"]


def test_read_file_sample_contract_is_complete():
    """The MCP-shaped contract has a name, description, and JSON schemas."""
    contract = READ_FILE_SAMPLE
    assert contract.name == "read_file_sample"
    assert contract.description
    assert contract.input_schema["type"] == "object"
    assert contract.input_schema["required"] == ["source_id"]
    assert "path" not in contract.input_schema["properties"]
    assert contract.result_schema["required"] == [
        "columns",
        "dtypes",
        "row_count",
        "rows",
    ]
