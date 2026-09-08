# sources.py

"""Data-source records identified by a stable hash of file bytes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DataSource:
    """One registered analysis source. `source_id` is derived from content."""

    source_id: str
    kind: Literal["file"]  # Widen in 1.15 when postgres sources are added.
    original_name: str
    stored_path: Path
    sha256: str
    created_at: datetime


def hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of the file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_source_id(sha256: str) -> str:
    """Build a stable file source id from a content hash."""
    return f"file-{sha256}"


def make_file_source(
    path: Path,
    *,
    original_name: str | None = None,
    stored_path: Path | None = None,
    created_at: datetime | None = None,
) -> DataSource:
    """Create a file source record from bytes at `path`."""
    digest = hash_file(path)
    return DataSource(
        source_id=file_source_id(digest),
        kind="file",
        original_name=original_name or path.name,
        stored_path=stored_path if stored_path is not None else path,
        sha256=digest,
        created_at=created_at or datetime.now(timezone.utc),
    )
