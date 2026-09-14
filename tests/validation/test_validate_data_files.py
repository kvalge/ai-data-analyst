# test_validate_data_files.py

"""Tests for analysis data-file input validation."""

from pathlib import Path

import pytest

from src.config import DEFAULT_MAX_UPLOAD_BYTES
from src.validation.data_files import FileValidationError, validate_data_file


def test_accepts_sample_sales_csv(sample_sales_csv: Path):
    """The committed sales fixture is a valid data file."""
    assert validate_data_file(sample_sales_csv, max_bytes=DEFAULT_MAX_UPLOAD_BYTES) == (
        sample_sales_csv.expanduser()
    )


def test_accepts_uppercase_csv_suffix(tmp_path: Path, sample_sales_csv: Path):
    """Allowlisted suffixes are matched case-insensitively."""
    path = tmp_path / "sales.CSV"
    path.write_bytes(sample_sales_csv.read_bytes())

    assert validate_data_file(path, max_bytes=DEFAULT_MAX_UPLOAD_BYTES) == path.expanduser()


def test_rejects_wrong_suffix(tmp_path: Path):
    """A non-allowlisted suffix is rejected."""
    path = tmp_path / "notes.txt"
    path.write_text("not a dataset", encoding="utf-8")

    with pytest.raises(FileValidationError, match="suffix") as exc_info:
        validate_data_file(path, max_bytes=DEFAULT_MAX_UPLOAD_BYTES)

    assert ".txt" in str(exc_info.value)


def test_rejects_empty_file(tmp_path: Path):
    """An allowlisted suffix with zero bytes is rejected."""
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")

    with pytest.raises(FileValidationError, match="empty"):
        validate_data_file(path, max_bytes=DEFAULT_MAX_UPLOAD_BYTES)


def test_rejects_over_max_bytes(tmp_path: Path):
    """A file larger than max_bytes is rejected."""
    path = tmp_path / "big.csv"
    path.write_bytes(b"a" * 11)

    with pytest.raises(FileValidationError, match="11") as exc_info:
        validate_data_file(path, max_bytes=10)

    assert "10" in str(exc_info.value)


def test_rejects_unreadable_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An unreadable file fails as FileValidationError, not PermissionError."""
    path = tmp_path / "locked.csv"
    path.write_text("date,region,revenue\n", encoding="utf-8")

    monkeypatch.setattr(
        "src.validation.uploads.os.access", lambda _path, _mode: False
    )

    with pytest.raises(FileValidationError, match="readable"):
        validate_data_file(path, max_bytes=DEFAULT_MAX_UPLOAD_BYTES)
