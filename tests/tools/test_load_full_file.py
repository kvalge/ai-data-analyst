# test_load_full_file.py

"""Tests for load_full_file under and over row/byte limits.

over_row_limit and over_row_limit_after_load exist so `reason` shows
the cheap-path skip versus the expensive post-load reject.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_settings
from src.storage.registry import save_file_source
from src.storage.sources import make_postgres_source
from src.tools.load_full_file import LOAD_FULL_FILE, load_full_file
from src.tools.profile_source import ProfileError
from src.validation.data_files import FileValidationError
from tests.tool_schema import assert_keys_match_required

_BIG_BYTES = 50 * 1024 * 1024


def _register_sales(tmp_path: Path, sample_sales_csv: Path):
    upload_dir = tmp_path / "uploads"
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(incoming, upload_dir, original_name="sales.csv")
    return upload_dir, saved


def test_load_full_file_under_limits(tmp_path: Path, sample_sales_csv: Path):
    """A small CSV under both caps loads metadata, not row records."""
    upload_dir, saved = _register_sales(tmp_path, sample_sales_csv)
    result = load_full_file(
        upload_dir=upload_dir,
        source_id=saved.source_id,
        max_full_load_rows=10,
        max_bytes=_BIG_BYTES,
    )
    assert result["status"] == "loaded"
    assert result["reason"] == ""
    assert result["row_count"] == 5
    assert result["columns"] == ["date", "region", "revenue"]
    assert "rows" not in result
    assert_keys_match_required(result, LOAD_FULL_FILE.result_schema)
    assert json.loads(json.dumps(result)) == result


# These two prove the cheap-path / expensive-path distinction is visible in reason.
def test_load_full_file_over_row_limit(
    tmp_path: Path, sample_sales_csv: Path, monkeypatch: pytest.MonkeyPatch
):
    """More CSV rows than the cap is needs_approval and does not pandas-load."""
    upload_dir, saved = _register_sales(tmp_path, sample_sales_csv)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("should not pandas-load over the row limit")

    monkeypatch.setattr("pandas.read_csv", boom)
    result = load_full_file(
        upload_dir=upload_dir,
        source_id=saved.source_id,
        max_full_load_rows=3,
        max_bytes=_BIG_BYTES,
    )
    assert result["status"] == "needs_approval"
    assert result["reason"] == "row_count"
    assert result["row_count"] == 5
    assert result["columns"] == []
    assert_keys_match_required(result, LOAD_FULL_FILE.result_schema)


def test_load_full_file_over_row_limit_after_load(tmp_path: Path):
    """JSON has no cheap row count, so over-cap is row_count_post_load."""
    upload_dir = tmp_path / "uploads"
    incoming = tmp_path / "sales.json"
    incoming.write_text(
        json.dumps(
            [
                {"date": "2024-01-01", "region": "North", "revenue": 120},
                {"date": "2024-01-02", "region": "South", "revenue": 95},
                {"date": "2024-01-03", "region": "North", "revenue": 130},
                {"date": "2024-01-04", "region": "East", "revenue": 80},
                {"date": "2024-01-05", "region": "West", "revenue": 150},
            ]
        ),
        encoding="utf-8",
    )
    saved = save_file_source(incoming, upload_dir, original_name="sales.json")
    result = load_full_file(
        upload_dir=upload_dir,
        source_id=saved.source_id,
        max_full_load_rows=3,
        max_bytes=_BIG_BYTES,
    )
    assert result["status"] == "needs_approval"
    assert result["reason"] == "row_count_post_load"
    assert result["row_count"] == 5
    assert result["columns"] == []
    assert "rows" not in result
    assert_keys_match_required(result, LOAD_FULL_FILE.result_schema)


def test_load_full_file_over_byte_limit(
    tmp_path: Path, sample_sales_csv: Path, monkeypatch: pytest.MonkeyPatch
):
    """A file larger than max_bytes is needs_approval and does not pandas-load."""
    upload_dir, saved = _register_sales(tmp_path, sample_sales_csv)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("should not pandas-load over the byte limit")

    monkeypatch.setattr("pandas.read_csv", boom)
    result = load_full_file(
        upload_dir=upload_dir,
        source_id=saved.source_id,
        max_full_load_rows=10_000,
        max_bytes=1,
    )
    assert result["status"] == "needs_approval"
    assert "size_bytes" in result["reason"]
    assert result["size_bytes"] > 1
    assert result["columns"] == []


def test_load_full_file_unknown_source_id(tmp_path: Path):
    """Unknown file ids are a domain error, not a silent skip."""
    with pytest.raises(FileValidationError, match="Unknown source_id"):
        load_full_file(
            upload_dir=tmp_path / "uploads",
            source_id="file-missing",
            max_full_load_rows=10,
            max_bytes=_BIG_BYTES,
        )


def test_load_full_file_rejects_postgres_kind(tmp_path: Path):
    """Postgres is not loaded as a file."""
    settings = load_settings(
        environ={
            "DB_HOST": "db.local",
            "DB_PORT": "5432",
            "DB_NAME": "analytics",
            "DB_USER": "reader",
            "DB_PASSWORD": "super-secret-password",
        },
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )
    source = make_postgres_source(settings)
    with pytest.raises(ProfileError, match="Postgres"):
        load_full_file(
            upload_dir=tmp_path / "uploads",
            source_id=source.source_id,
            max_full_load_rows=10,
            max_bytes=_BIG_BYTES,
            settings=settings,
        )


def test_load_full_file_contract_is_complete():
    """The MCP-shaped contract has a name, description, and JSON schemas."""
    contract = LOAD_FULL_FILE
    assert contract.name == "load_full_file"
    assert contract.description
    assert contract.input_schema["required"] == ["source_id"]
    assert contract.result_schema["required"] == [
        "status",
        "source_id",
        "row_count",
        "size_bytes",
        "max_full_load_rows",
        "max_bytes",
        "reason",
        "columns",
    ]
