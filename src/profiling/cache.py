# cache.py

"""JSON profile cache under CACHE_DIR. Invalidate when the content hash changes."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.storage.sources import DataSource

_LOG = logging.getLogger(__name__)

_SAFE_SOURCE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_CACHE_TMP_SUFFIX = ".tmp"
_ENVELOPE_SOURCE_ID = "source_id"
_ENVELOPE_HASH = "content_hash"
_ENVELOPE_PAYLOAD = "payload"


def read_profile_cache(cache_dir: Path, source: DataSource) -> dict[str, Any] | None:
    """Return the cached payload on hit. Miss or stale hash returns None."""
    path = _cache_path(cache_dir, source.source_id)
    if path is None or not path.is_file():
        _LOG.info("profile cache miss source_id=%s", source.source_id)
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _LOG.warning("profile cache corrupt source_id=%s", source.source_id)
        return None
    if not isinstance(raw, dict):
        return None
    stored_hash = raw.get(_ENVELOPE_HASH)
    payload = raw.get(_ENVELOPE_PAYLOAD)
    # TODO: Postgres sha256 is a connection fingerprint, not table contents;
    # revisit invalidation in 2.11 when profile_source supports postgres.
    if stored_hash != source.sha256 or not isinstance(payload, dict):
        _LOG.info(
            "profile cache stale source_id=%s stored_hash=%s current_hash=%s",
            source.source_id,
            stored_hash,
            source.sha256,
        )
        return None
    _LOG.info("profile cache hit source_id=%s", source.source_id)
    return payload


def write_profile_cache(
    cache_dir: Path, source: DataSource, payload: dict[str, Any]
) -> Path:
    """Write `payload` keyed by source_id and content hash. Atomic replace."""
    path = _cache_path(cache_dir, source.source_id)
    if path is None:
        raise ValueError(f"Unsafe source_id for cache filename: {source.source_id!r}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    envelope = {
        _ENVELOPE_SOURCE_ID: source.source_id,
        _ENVELOPE_HASH: source.sha256,
        _ENVELOPE_PAYLOAD: payload,
    }
    tmp_path = path.with_suffix(path.suffix + _CACHE_TMP_SUFFIX)
    tmp_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    _LOG.info("profile cache write source_id=%s", source.source_id)
    return path


def _cache_path(cache_dir: Path, source_id: str) -> Path | None:
    if not _SAFE_SOURCE_ID.fullmatch(source_id):
        return None
    return (cache_dir / f"{source_id}.json").resolve()
