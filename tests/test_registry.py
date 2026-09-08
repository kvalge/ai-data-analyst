# test_registry.py

"""Tests for persisting file sources and re-listing from registry.json."""

from pathlib import Path

import pytest

from src.storage.registry import (
    RegistryError,
    list_file_sources,
    registry_path,
    save_file_source,
)


def test_list_file_sources_empty_when_missing(tmp_path: Path):
    """A directory with no registry.json lists no sources."""
    assert list_file_sources(tmp_path) == []


def test_save_round_trips_through_registry(tmp_path: Path, sample_sales_csv: Path):
    """Save then list returns the same DataSource, including stored path."""
    upload_dir = tmp_path / "uploads"
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())

    saved = save_file_source(incoming, upload_dir, original_name="sales.csv")
    listed = list_file_sources(upload_dir)

    assert listed == [saved]
    assert saved.stored_path.parent == upload_dir.resolve()
    assert saved.stored_path.name.startswith("file-")
    assert saved.stored_path.suffix == ".csv"
    assert saved.stored_path.is_file()
    assert saved.stored_path.read_bytes() == sample_sales_csv.read_bytes()


def test_second_save_same_bytes_is_noop(tmp_path: Path, sample_sales_csv: Path):
    """Re-uploading the same bytes does not recopy or change created_at."""
    upload_dir = tmp_path / "uploads"
    first_src = tmp_path / "a.csv"
    second_src = tmp_path / "b.csv"
    payload = sample_sales_csv.read_bytes()
    first_src.write_bytes(payload)
    second_src.write_bytes(payload)

    first = save_file_source(first_src, upload_dir, original_name="a.csv")
    mtime_ns = first.stored_path.stat().st_mtime_ns
    second = save_file_source(second_src, upload_dir, original_name="b.csv")

    assert second == first
    assert second.created_at == first.created_at
    assert second.original_name == "a.csv"
    assert len(list_file_sources(upload_dir)) == 1
    assert first.stored_path.stat().st_mtime_ns == mtime_ns


def test_different_files_are_both_listed(tmp_path: Path, sample_sales_csv: Path):
    """Two different payloads produce two registry entries."""
    upload_dir = tmp_path / "uploads"
    first_src = tmp_path / "a.csv"
    second_src = tmp_path / "b.csv"
    first_src.write_bytes(sample_sales_csv.read_bytes())
    second_src.write_bytes(sample_sales_csv.read_bytes() + b"\n")

    save_file_source(first_src, upload_dir)
    save_file_source(second_src, upload_dir)

    assert len(list_file_sources(upload_dir)) == 2


def test_stored_suffix_is_lowercase(tmp_path: Path, sample_sales_csv: Path):
    """Stored filenames use a lowercase suffix even if the original name does not."""
    upload_dir = tmp_path / "uploads"
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())

    saved = save_file_source(incoming, upload_dir, original_name="Report.CSV")

    assert saved.stored_path.suffix == ".csv"
    assert saved.original_name == "Report.CSV"


def test_list_file_sources_rejects_corrupt_json(tmp_path: Path):
    """Corrupt registry JSON raises RegistryError, not a raw JSONDecodeError."""
    registry_path(tmp_path).write_text("{not-json", encoding="utf-8")

    with pytest.raises(RegistryError, match="Invalid source registry JSON"):
        list_file_sources(tmp_path)


def test_list_file_sources_rejects_incomplete_entry(tmp_path: Path):
    """A source object missing required fields raises RegistryError."""
    registry_path(tmp_path).write_text(
        '{"sources": [{"kind": "file", "source_id": "file-x"}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match="incomplete"):
        list_file_sources(tmp_path)


def test_list_file_sources_rejects_unsupported_kind(tmp_path: Path):
    """Postgres is env-backed; a postgres row in registry.json is invalid."""
    registry_path(tmp_path).write_text(
        """
        {
          "sources": [
            {
              "source_id": "pg-1",
              "kind": "postgres",
              "original_name": "orders",
              "stored_name": "unused.csv",
              "sha256": "abc",
              "created_at": "2026-01-01T00:00:00+00:00"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match="Unsupported source kind"):
        list_file_sources(tmp_path)
