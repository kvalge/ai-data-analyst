# test_validate_context_files.py

"""Tests for business-context file input validation."""

from pathlib import Path

import pytest

from src.config import DEFAULT_MAX_UPLOAD_BYTES
from src.validation.context_files import validate_context_file
from src.validation.uploads import FileValidationError


def test_accepts_sample_glossary_md(sample_glossary_md: Path):
    """The committed glossary fixture is a valid context file."""
    assert validate_context_file(sample_glossary_md, max_bytes=DEFAULT_MAX_UPLOAD_BYTES) == (
        sample_glossary_md.expanduser()
    )


def test_accepts_uppercase_md_suffix(tmp_path: Path, sample_glossary_md: Path):
    """Allowlisted suffixes are matched case-insensitively."""
    path = tmp_path / "glossary.MD"
    path.write_bytes(sample_glossary_md.read_bytes())

    assert validate_context_file(path, max_bytes=DEFAULT_MAX_UPLOAD_BYTES) == path.expanduser()


def test_accepts_txt_and_pdf_suffixes(tmp_path: Path):
    """Text and PDF context documents are allowlisted."""
    notes = tmp_path / "notes.txt"
    notes.write_text("KPI: revenue is net of returns.\n", encoding="utf-8")
    pdf = tmp_path / "notes.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    assert validate_context_file(notes, max_bytes=DEFAULT_MAX_UPLOAD_BYTES) == notes.expanduser()
    assert validate_context_file(pdf, max_bytes=DEFAULT_MAX_UPLOAD_BYTES) == pdf.expanduser()


def test_rejects_csv_suffix(tmp_path: Path, sample_sales_csv: Path):
    """A data-file suffix is not a context document."""
    path = tmp_path / "sales.csv"
    path.write_bytes(sample_sales_csv.read_bytes())

    with pytest.raises(FileValidationError, match="suffix") as exc_info:
        validate_context_file(path, max_bytes=DEFAULT_MAX_UPLOAD_BYTES)

    assert ".csv" in str(exc_info.value)


def test_rejects_empty_file(tmp_path: Path):
    """An allowlisted suffix with zero bytes is rejected."""
    path = tmp_path / "empty.md"
    path.write_bytes(b"")

    with pytest.raises(FileValidationError, match="empty"):
        validate_context_file(path, max_bytes=DEFAULT_MAX_UPLOAD_BYTES)


def test_rejects_over_max_bytes(tmp_path: Path):
    """A file larger than max_bytes is rejected."""
    path = tmp_path / "big.md"
    path.write_bytes(b"a" * 11)

    with pytest.raises(FileValidationError, match="11") as exc_info:
        validate_context_file(path, max_bytes=10)

    assert "10" in str(exc_info.value)


def test_rejects_unreadable_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An unreadable file fails as FileValidationError, not PermissionError."""
    path = tmp_path / "locked.md"
    path.write_text("# notes\n", encoding="utf-8")

    monkeypatch.setattr(
        "src.validation.uploads.os.access", lambda _path, _mode: False
    )

    with pytest.raises(FileValidationError, match="readable"):
        validate_context_file(path, max_bytes=DEFAULT_MAX_UPLOAD_BYTES)
