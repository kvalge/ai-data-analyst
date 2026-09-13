# test_index.py

"""Tests for persisted context reindex. No dataset rows; text only."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.rag.index import CONTEXT_INDEX_NAME, reindex_context
from src.storage.context import ingest_context_upload

_MAX = 50 * 1024 * 1024


def test_reindex_after_upload_hits_glossary(
    tmp_path: Path, sample_glossary_md: Path
):
    """Uploading the glossary fixture writes an index that contains it."""
    context_dir = tmp_path / "context"
    cache_dir = tmp_path / "cache"
    ingest_context_upload(
        sample_glossary_md.read_bytes(),
        "sample_glossary.md",
        context_dir,
        max_bytes=_MAX,
    )
    index = reindex_context(
        context_dir, cache_dir=cache_dir, max_bytes=_MAX
    )
    assert (cache_dir / CONTEXT_INDEX_NAME).is_file()
    assert any(
        "Revenue is net of returns." in chunk["text"] for chunk in index.chunks
    )
    assert all(chunk["source_file"] != "sales.csv" for chunk in index.chunks)


def test_reindex_is_idempotent_when_files_unchanged(
    tmp_path: Path, sample_glossary_md: Path, monkeypatch: pytest.MonkeyPatch
):
    """A second reindex with the same files does not reread or rehash them."""
    context_dir = tmp_path / "context"
    cache_dir = tmp_path / "cache"
    ingest_context_upload(
        sample_glossary_md.read_bytes(),
        "sample_glossary.md",
        context_dir,
        max_bytes=_MAX,
    )
    first = reindex_context(
        context_dir, cache_dir=cache_dir, max_bytes=_MAX
    )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("should not rebuild or rehash")

    monkeypatch.setattr("src.rag.index.build_context_index", boom)
    monkeypatch.setattr("src.rag.index.hash_file", boom)
    second = reindex_context(
        context_dir, cache_dir=cache_dir, max_bytes=_MAX
    )
    assert second.chunks == first.chunks


def test_reindex_skips_rebuild_when_only_mtime_changes(
    tmp_path: Path, sample_glossary_md: Path, monkeypatch: pytest.MonkeyPatch
):
    """A touch that keeps the same bytes does not rebuild the index."""
    context_dir = tmp_path / "context"
    cache_dir = tmp_path / "cache"
    ingest_context_upload(
        sample_glossary_md.read_bytes(),
        "sample_glossary.md",
        context_dir,
        max_bytes=_MAX,
    )
    first = reindex_context(
        context_dir, cache_dir=cache_dir, max_bytes=_MAX
    )
    stored = context_dir / "sample_glossary.md"
    stat = stored.stat()
    os.utime(stored, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("should not rebuild")

    monkeypatch.setattr("src.rag.index.build_context_index", boom)
    second = reindex_context(
        context_dir, cache_dir=cache_dir, max_bytes=_MAX
    )
    assert second.chunks == first.chunks


def test_reindex_rebuilds_after_overwrite(tmp_path: Path):
    """A same-name upload replaces the indexed text."""
    context_dir = tmp_path / "context"
    cache_dir = tmp_path / "cache"
    ingest_context_upload(
        b"# Sales glossary\n\nRevenue is net of returns.\n",
        "glossary.md",
        context_dir,
        max_bytes=_MAX,
    )
    first = reindex_context(
        context_dir, cache_dir=cache_dir, max_bytes=_MAX
    )
    ingest_context_upload(
        b"# Shipping\n\nLead time is five days.\n",
        "glossary.md",
        context_dir,
        max_bytes=_MAX,
    )
    second = reindex_context(
        context_dir, cache_dir=cache_dir, max_bytes=_MAX
    )
    blob = " ".join(chunk["text"] for chunk in second.chunks)
    assert "Lead time is five days." in blob
    assert "Revenue is net of returns." not in blob
    assert first.chunks != second.chunks
