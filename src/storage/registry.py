# registry.py

"""Persist file sources in UPLOAD_DIR with a single registry.json."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from src.storage.sources import DataSource, make_file_source

REGISTRY_NAME = "registry.json"
_REGISTRY_TMP_NAME = "registry.json.tmp"


class RegistryError(ValueError):
    """The source registry on disk is missing or invalid."""


def registry_path(upload_dir: Path) -> Path:
    """Return the registry.json path inside `upload_dir`."""
    return upload_dir / REGISTRY_NAME


def list_file_sources(upload_dir: Path) -> list[DataSource]:
    """Load file sources from registry.json. Missing file means none yet."""
    path = registry_path(upload_dir)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Invalid source registry JSON: {path}") from exc
    rows = _registry_rows(payload, path)
    return [_source_from_row(row, upload_dir, path) for row in rows]


def get_file_source(upload_dir: Path, source_id: str) -> DataSource | None:
    """Return the registered file source with `source_id`, or None."""
    for source in list_file_sources(upload_dir):
        if source.source_id == source_id:
            return source
    return None


def save_file_source(
    path: Path,
    upload_dir: Path,
    *,
    original_name: str | None = None,
) -> DataSource:
    """Copy `path` into `upload_dir` and register it. Same bytes are a no-op."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    incoming = make_file_source(path, original_name=original_name)
    sources = list_file_sources(upload_dir)
    for existing in sources:
        if existing.source_id == incoming.source_id:
            return existing

    suffix = Path(incoming.original_name).suffix.lower()
    stored_name = f"{incoming.source_id}{suffix}"
    stored_path = (upload_dir / stored_name).resolve()
    if path.resolve() != stored_path:
        shutil.copyfile(path, stored_path)

    saved = DataSource(
        source_id=incoming.source_id,
        kind="file",
        original_name=incoming.original_name,
        stored_path=stored_path,
        sha256=incoming.sha256,
        created_at=incoming.created_at,
    )
    sources.append(saved)
    _write_registry(upload_dir, sources)
    return saved


def _registry_rows(payload: object, path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or "sources" not in payload:
        raise RegistryError(f"Source registry must be an object with 'sources': {path}")
    rows = payload["sources"]
    if not isinstance(rows, list):
        raise RegistryError(f"Source registry 'sources' must be a list: {path}")
    return rows


def _source_from_row(row: object, upload_dir: Path, path: Path) -> DataSource:
    if not isinstance(row, dict):
        raise RegistryError(f"Source registry entries must be objects: {path}")
    try:
        source_id = str(row["source_id"])
        original_name = str(row["original_name"])
        stored_name = str(row["stored_name"])
        sha256 = str(row["sha256"])
        created_at = datetime.fromisoformat(str(row["created_at"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RegistryError(f"Source registry entry is incomplete: {path}") from exc
    if row.get("kind") != "file":
        # Postgres is env-backed (1.15); it is not persisted in registry.json.
        raise RegistryError(f"Unsupported source kind in registry: {path}")
    stored_path = (upload_dir / stored_name).resolve()
    # TODO: a registry row whose stored file was deleted is still listed (known gap).
    return DataSource(
        source_id=source_id,
        kind="file",
        original_name=original_name,
        stored_path=stored_path,
        sha256=sha256,
        created_at=created_at,
    )


def _write_registry(upload_dir: Path, sources: list[DataSource]) -> None:
    """Write registry.json via a temp file so a crash does not leave partial JSON."""
    payload = {
        "sources": [
            {
                "source_id": source.source_id,
                "kind": source.kind,
                "original_name": source.original_name,
                "stored_name": source.stored_path.name if source.stored_path else "",
                "sha256": source.sha256,
                "created_at": source.created_at.isoformat(),
            }
            for source in sources
        ]
    }
    tmp_path = upload_dir / _REGISTRY_TMP_NAME
    tmp_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(registry_path(upload_dir))
