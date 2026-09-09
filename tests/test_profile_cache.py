# test_profile_cache.py

"""Tests for the profile JSON cache (hit, miss, stale hash)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.profiling.cache import read_profile_cache, write_profile_cache
from src.storage.sources import DataSource


def _source(*, source_id: str, sha256: str) -> DataSource:
    return DataSource(
        source_id=source_id,
        kind="file",
        original_name="sales.csv",
        stored_path=None,
        sha256=sha256,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_profile_cache_miss_when_empty(tmp_path: Path):
    """No cache file is a miss."""
    source = _source(source_id="file-aaa", sha256="aaa")
    assert read_profile_cache(tmp_path / "cache", source) is None


def test_profile_cache_hit_returns_payload(tmp_path: Path):
    """A matching source_id and hash returns the stored payload."""
    cache_dir = tmp_path / "cache"
    source = _source(source_id="file-aaa", sha256="aaa")
    payload = {"columns": ["revenue"], "file_row_count": 5}

    write_profile_cache(cache_dir, source, payload)
    assert read_profile_cache(cache_dir, source) == payload


def test_profile_cache_stale_hash_is_miss(tmp_path: Path):
    """Same source_id with a different content hash does not return the old payload."""
    cache_dir = tmp_path / "cache"
    fresh = _source(source_id="file-aaa", sha256="aaa")
    write_profile_cache(cache_dir, fresh, {"columns": ["revenue"]})

    stale = _source(source_id="file-aaa", sha256="bbb")
    assert read_profile_cache(cache_dir, stale) is None
    assert read_profile_cache(cache_dir, fresh) == {"columns": ["revenue"]}


def test_profile_cache_corrupt_json_is_miss(tmp_path: Path):
    """Unreadable cache JSON is a miss, not a crash."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "file-aaa.json").write_text("{not-json", encoding="utf-8")
    source = _source(source_id="file-aaa", sha256="aaa")
    assert read_profile_cache(cache_dir, source) is None
