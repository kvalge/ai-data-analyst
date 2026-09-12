# test_sources.py

"""Tests for file source identity and hashing."""

from pathlib import Path

from src.storage.sources import file_source_id, make_file_source


def test_same_bytes_same_source_id(tmp_path: Path, sample_sales_csv: Path):
    """Two copies of the same bytes share source_id, regardless of filename."""
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    payload = sample_sales_csv.read_bytes()
    first.write_bytes(payload)
    second.write_bytes(payload)

    left = make_file_source(first)
    right = make_file_source(second)

    assert left.source_id == right.source_id
    assert left.sha256 == right.sha256
    assert left.source_id == file_source_id(left.sha256)
    assert left.kind == "file"
    assert left.original_name == "a.csv"
    assert right.original_name == "b.csv"


def test_different_bytes_different_source_id(tmp_path: Path, sample_sales_csv: Path):
    """Changing file bytes changes source_id."""
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    first.write_bytes(sample_sales_csv.read_bytes())
    second.write_bytes(sample_sales_csv.read_bytes() + b"\n")

    assert make_file_source(first).source_id != make_file_source(second).source_id
