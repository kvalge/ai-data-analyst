# sources.py

"""Data-source records identified by file hash or a Postgres fingerprint."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.config import Settings

_CHUNK_SIZE = 1024 * 1024

SourceKind = Literal["file", "postgres"]


@dataclass(frozen=True)
class DataSource:
    """One analysis source. File ids come from bytes; Postgres from env (no password)."""

    source_id: str
    kind: SourceKind
    original_name: str
    stored_path: Path | None
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


def postgres_fingerprint(settings: Settings) -> str:
    """Hash host/port/name/user. Password is not part of the identity."""
    material = (
        f"{settings.db_host}\0{settings.db_port}\0"
        f"{settings.db_name}\0{settings.db_user}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def postgres_source_id(settings: Settings) -> str:
    """Build a stable Postgres source id from the connection fingerprint."""
    return f"postgres-{postgres_fingerprint(settings)}"


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


def make_postgres_source(
    settings: Settings,
    *,
    created_at: datetime | None = None,
) -> DataSource:
    """Create a Postgres source from settings. Password is not stored on the record."""
    digest = postgres_fingerprint(settings)
    return DataSource(
        source_id=postgres_source_id(settings),
        kind="postgres",
        original_name=(
            f"{settings.db_user}@{settings.db_host}:{settings.db_port}/"
            f"{settings.db_name}"
        ),
        stored_path=None,
        sha256=digest,
        # Listing time, not first use. Do not rely on this for stability.
        created_at=created_at or datetime.now(timezone.utc),
    )
