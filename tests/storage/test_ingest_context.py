# test_ingest_context.py

"""Tests for context-file ingest (validate then copy into CONTEXT_DIR)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.config import DEFAULT_MAX_UPLOAD_BYTES
from src.storage.context import ingest_context_upload, list_context_files
from src.tools.list_sources import list_available_sources
from src.validation.uploads import FileValidationError


def test_ingest_markdown_is_context_not_data_source(
    tmp_path: Path, sample_glossary_md: Path
):
    """A .md file is stored under context and is not a data source."""
    context_dir = tmp_path / "context"
    upload_dir = tmp_path / "uploads"
    payload = sample_glossary_md.read_bytes()

    saved = ingest_context_upload(
        payload, "glossary.md", context_dir, max_bytes=DEFAULT_MAX_UPLOAD_BYTES
    )

    assert saved == context_dir / "glossary.md"
    assert saved.is_file()
    assert list_context_files(context_dir) == [saved]
    assert list_available_sources(upload_dir) == {"sources": []}


def test_ingest_context_rejects_csv(tmp_path: Path, sample_sales_csv: Path):
    """A CSV is not stored as context."""
    context_dir = tmp_path / "context"
    payload = sample_sales_csv.read_bytes()

    with pytest.raises(FileValidationError, match="suffix"):
        ingest_context_upload(
            payload, "sales.csv", context_dir, max_bytes=DEFAULT_MAX_UPLOAD_BYTES
        )

    assert list_context_files(context_dir) == []


def test_ingest_context_keeps_basename_only(tmp_path: Path, sample_glossary_md: Path):
    """Path components in the original name do not escape CONTEXT_DIR."""
    context_dir = tmp_path / "context"
    payload = sample_glossary_md.read_bytes()

    saved = ingest_context_upload(
        payload, "../glossary.md", context_dir, max_bytes=DEFAULT_MAX_UPLOAD_BYTES
    )

    assert saved == context_dir / "glossary.md"
    assert saved.is_file()
    assert not (tmp_path / "glossary.md").exists()


def test_ingest_context_same_name_overwrites(tmp_path: Path):
    """A second upload with the same basename replaces the stored file."""
    context_dir = tmp_path / "context"

    first = ingest_context_upload(
        b"# old glossary\n", "glossary.md", context_dir, max_bytes=DEFAULT_MAX_UPLOAD_BYTES
    )
    saved = ingest_context_upload(
        b"# new glossary\n", "glossary.md", context_dir, max_bytes=DEFAULT_MAX_UPLOAD_BYTES
    )

    assert saved == first
    assert saved.read_text(encoding="utf-8") == "# new glossary\n"
    assert list_context_files(context_dir) == [saved]


def test_ingest_context_wraps_copy_oserror(
    tmp_path: Path, sample_glossary_md: Path, monkeypatch: pytest.MonkeyPatch
):
    """A disk write failure becomes FileValidationError, not a raw OSError."""
    created: list[Path] = []
    real_named = tempfile.NamedTemporaryFile

    def tracking_named(*args: object, **kwargs: object):
        handle = real_named(*args, **kwargs)
        created.append(Path(handle.name))
        return handle

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("src.storage.context.tempfile.NamedTemporaryFile", tracking_named)
    monkeypatch.setattr("src.storage.context.shutil.copyfile", boom)

    with pytest.raises(FileValidationError, match="Could not save"):
        ingest_context_upload(
            sample_glossary_md.read_bytes(),
            "glossary.md",
            tmp_path / "context",
            max_bytes=DEFAULT_MAX_UPLOAD_BYTES,
        )

    assert len(created) == 1
    assert not created[0].exists()
