# test_profile_source.py

"""Tests for profile_source (fresh vs cached compact summaries)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_settings
from src.profiling.dq import NULLS_RESULT_KEYS
from src.profiling.schema import SCHEMA_RESULT_KEYS
from src.storage.registry import save_file_source
from src.storage.sources import make_postgres_source
from src.tools.profile_source import PROFILE_SOURCE, ProfileError, profile_source
from src.validation.data_files import FileValidationError
from tests.tool_schema import assert_keys_match_required

_MAX = 50 * 1024 * 1024


def _register_sales(tmp_path: Path, sample_sales_csv: Path):
    upload_dir = tmp_path / "uploads"
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(incoming, upload_dir, original_name="sales.csv")
    return upload_dir, tmp_path / "cache", saved


def test_profile_source_fresh_then_cached(tmp_path: Path, sample_sales_csv: Path):
    """First call computes; second call is a cache hit with the same summary."""
    upload_dir, cache_dir, saved = _register_sales(tmp_path, sample_sales_csv)

    fresh = profile_source(
        upload_dir=upload_dir,
        cache_dir=cache_dir,
        n_rows=50,
        max_bytes=_MAX,
        source_id=saved.source_id,
    )
    cached = profile_source(
        upload_dir=upload_dir,
        cache_dir=cache_dir,
        n_rows=50,
        max_bytes=_MAX,
        source_id=saved.source_id,
    )

    assert fresh["cached"] is False
    assert cached["cached"] is True
    assert "rows" not in fresh
    assert_keys_match_required(fresh, PROFILE_SOURCE.result_schema)
    assert_keys_match_required(cached, PROFILE_SOURCE.result_schema)
    core_fresh = {key: value for key, value in fresh.items() if key != "cached"}
    core_cached = {key: value for key, value in cached.items() if key != "cached"}
    assert core_fresh == core_cached
    assert fresh["source_id"] == saved.source_id
    assert fresh["sample_row_count"] == 5
    assert set(fresh["schema"]) == SCHEMA_RESULT_KEYS
    assert fresh["schema"]["file_row_count"] == 5
    assert set(fresh["dq"]["nulls"]) == NULLS_RESULT_KEYS
    assert fresh["dq"]["duplicates"]["duplicate_row_count"] == 0
    assert fresh["eda"]["correlations"]["skipped"] is True
    encoded = json.dumps(fresh)
    assert json.loads(encoded) == fresh


def test_profile_source_bounded_read_not_full_file(
    tmp_path: Path, sample_sales_csv: Path
):
    """n_rows caps the profile frame; cheap file_row_count still sees the CSV."""
    upload_dir, cache_dir, saved = _register_sales(tmp_path, sample_sales_csv)
    result = profile_source(
        upload_dir=upload_dir,
        cache_dir=cache_dir,
        n_rows=2,
        max_bytes=_MAX,
        source_id=saved.source_id,
    )
    assert result["sample_row_count"] == 2
    assert result["schema"]["sample_row_count"] == 2
    assert result["schema"]["file_row_count"] == 5


def test_profile_source_unknown_source_id(tmp_path: Path):
    """Unknown file ids are a domain error, not an empty profile."""
    with pytest.raises(FileValidationError, match="Unknown source_id"):
        profile_source(
            upload_dir=tmp_path / "uploads",
            cache_dir=tmp_path / "cache",
            n_rows=2,
            max_bytes=_MAX,
            source_id="file-missing",
        )


def test_profile_source_rejects_postgres_kind(tmp_path: Path):
    """A resolved Postgres DataSource is not profiled in 2.11."""
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
        profile_source(
            upload_dir=tmp_path / "uploads",
            cache_dir=tmp_path / "cache",
            n_rows=2,
            max_bytes=_MAX,
            source_id=source.source_id,
            settings=settings,
        )


def test_profile_source_rejects_n_rows_below_one(
    tmp_path: Path, sample_sales_csv: Path
):
    """n_rows must be a positive count, even when the source exists."""
    upload_dir, cache_dir, saved = _register_sales(tmp_path, sample_sales_csv)
    with pytest.raises(FileValidationError, match="n_rows"):
        profile_source(
            upload_dir=upload_dir,
            cache_dir=cache_dir,
            n_rows=0,
            max_bytes=_MAX,
            source_id=saved.source_id,
        )


def test_profile_source_contract_is_complete():
    """The MCP-shaped contract has a name, description, and JSON schemas."""
    contract = PROFILE_SOURCE
    assert contract.name == "profile_source"
    assert contract.description
    assert contract.input_schema["required"] == ["source_id"]
    assert contract.result_schema["required"] == [
        "source_id",
        "cached",
        "sample_row_count",
        "schema",
        "dq",
        "eda",
    ]
