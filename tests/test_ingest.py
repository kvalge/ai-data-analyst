# test_ingest.py

"""Tests for data-file ingest (validate then persist)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.storage.ingest import ingest_data_upload
from src.storage.registry import RegistryError
from src.tools.list_sources import list_available_sources
from src.validation.data_files import FileValidationError


def test_ingest_sample_csv_lists_one_source(tmp_path: Path, sample_sales_csv: Path):
    """A valid CSV is registered and appears in list_available_sources."""
    upload_dir = tmp_path / "uploads"
    payload = sample_sales_csv.read_bytes()

    saved = ingest_data_upload(
        payload, "sales.csv", upload_dir, max_bytes=50 * 1024 * 1024
    )

    listed = list_available_sources(upload_dir)
    assert len(listed["sources"]) == 1
    assert listed["sources"][0]["original_name"] == "sales.csv"
    assert listed["sources"][0]["source_id"] == saved.source_id


def test_ingest_rejects_wrong_suffix(tmp_path: Path):
    """A .txt upload fails validation and is not registered."""
    upload_dir = tmp_path / "uploads"

    with pytest.raises(FileValidationError, match="suffix"):
        ingest_data_upload(
            b"not a dataset", "notes.txt", upload_dir, max_bytes=50 * 1024 * 1024
        )

    assert list_available_sources(upload_dir) == {"sources": []}


def test_ingest_wraps_save_oserror(
    tmp_path: Path, sample_sales_csv: Path, monkeypatch: pytest.MonkeyPatch
):
    """A disk write failure becomes FileValidationError, not a raw OSError."""
    _assert_save_failure_cleans_temp(
        tmp_path,
        sample_sales_csv.read_bytes(),
        monkeypatch,
        error=OSError("disk full"),
        expected=FileValidationError,
        match="Could not save",
    )


def test_ingest_propagates_registry_error(
    tmp_path: Path, sample_sales_csv: Path, monkeypatch: pytest.MonkeyPatch
):
    """A corrupt registry stays RegistryError so the UI can show that message."""
    _assert_save_failure_cleans_temp(
        tmp_path,
        sample_sales_csv.read_bytes(),
        monkeypatch,
        error=RegistryError("Invalid source registry JSON: uploads/registry.json"),
        expected=RegistryError,
        match="Invalid source registry JSON",
    )


def _assert_save_failure_cleans_temp(
    tmp_path: Path,
    payload: bytes,
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: BaseException,
    expected: type[BaseException],
    match: str,
) -> None:
    """Run ingest with a failing save and assert the staging temp file is gone."""
    created: list[Path] = []
    real_named = tempfile.NamedTemporaryFile

    def tracking_named(*args: object, **kwargs: object):
        handle = real_named(*args, **kwargs)
        created.append(Path(handle.name))
        return handle

    def boom(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("src.storage.ingest.tempfile.NamedTemporaryFile", tracking_named)
    monkeypatch.setattr("src.storage.ingest.save_file_source", boom)

    with pytest.raises(expected, match=match):
        ingest_data_upload(
            payload, "sales.csv", tmp_path / "uploads", max_bytes=50 * 1024 * 1024
        )

    assert len(created) == 1
    assert not created[0].exists()
