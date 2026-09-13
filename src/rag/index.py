# index.py

"""Build and persist the context index. Rebuild when context files change."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.rag.chunker import chunk_context
from src.rag.readers import read_context_file
from src.rag.retrieve import DomainContextIndex
from src.storage.context import list_context_files
from src.storage.sources import hash_file

_LOG = logging.getLogger(__name__)

CONTEXT_INDEX_NAME = "context_index.json"
_INDEX_TMP_SUFFIX = ".tmp"
_ENVELOPE_FILES = "files"
_ENVELOPE_CHUNKS = "chunks"


def build_context_index(context_dir: Path, *, max_bytes: int) -> DomainContextIndex:
    """Read and chunk allowlisted files in `context_dir`. Never data files."""
    chunks: list[dict[str, str]] = []
    files = list_context_files(context_dir)
    for path in files:
        document = read_context_file(path, max_bytes=max_bytes)
        chunks.extend(chunk_context(document))
    _LOG.info("build context index files=%s chunks=%s", len(files), len(chunks))
    return DomainContextIndex(chunks)


def reindex_context(
    context_dir: Path, *, cache_dir: Path, max_bytes: int
) -> DomainContextIndex:
    """Rebuild the persisted index when context files change. Idempotent.

    Compare name + size + mtime first. Content-hash only when those
    change. Does not log document text.
    """
    files = list_context_files(context_dir)
    stats = _file_stats(files)
    stored = _read_persisted_index(cache_dir)
    if stored is not None and _stats_match(stats, stored[0]):
        chunks = stored[1]
        _LOG.info(
            "context index unchanged files=%s chunks=%s",
            len(files),
            len(chunks),
        )
        return DomainContextIndex(chunks)
    digests = {path.name: hash_file(path) for path, _name, _size, _mtime in stats}
    records = _file_records(stats, digests)
    if stored is not None and _digests_match(records, stored[0]):
        _write_persisted_index(cache_dir, records, stored[1])
        _LOG.info(
            "context index unchanged files=%s chunks=%s",
            len(files),
            len(stored[1]),
        )
        return DomainContextIndex(stored[1])
    index = build_context_index(context_dir, max_bytes=max_bytes)
    _write_persisted_index(cache_dir, records, index.chunks)
    _LOG.info(
        "context index rebuilt files=%s chunks=%s",
        len(files),
        len(index.chunks),
    )
    return index


def _file_stats(
    files: Sequence[Path],
) -> list[tuple[Path, str, int, int]]:
    """Basename, size, and mtime for each allowlisted file. No content read."""
    rows: list[tuple[Path, str, int, int]] = []
    for path in files:
        stat = path.stat()
        rows.append(
            (path, path.name, int(stat.st_size), int(stat.st_mtime_ns))
        )
    return rows


def _file_records(
    stats: Sequence[tuple[Path, str, int, int]],
    digests: Mapping[str, str],
) -> list[dict[str, str | int]]:
    """Persistable per-file stamps. `digests` is already computed."""
    return [
        {
            "name": name,
            "size": size,
            "mtime_ns": mtime_ns,
            "sha256": digests[name],
        }
        for _path, name, size, mtime_ns in stats
    ]


def _stats_match(
    stats: Sequence[tuple[Path, str, int, int]],
    stored: Sequence[Mapping[str, str | int]],
) -> bool:
    """True when names, sizes, and mtimes match the persisted stamps."""
    if len(stats) != len(stored):
        return False
    for (_path, name, size, mtime_ns), record in zip(stats, stored):
        if (
            name != record.get("name")
            or size != record.get("size")
            or mtime_ns != record.get("mtime_ns")
        ):
            return False
    return True


def _digests_match(
    records: Sequence[Mapping[str, str | int]],
    stored: Sequence[Mapping[str, str | int]],
) -> bool:
    """True when names and content hashes match. Ignores mtime and size."""
    if len(records) != len(stored):
        return False
    for record, prior in zip(records, stored):
        if record.get("name") != prior.get("name"):
            return False
        if record.get("sha256") != prior.get("sha256"):
            return False
    return True


def _index_path(cache_dir: Path) -> Path:
    return cache_dir / CONTEXT_INDEX_NAME


def _read_persisted_index(
    cache_dir: Path,
) -> tuple[list[dict[str, str | int]], list[dict[str, str]]] | None:
    """Return `(file stamps, chunks)` or None when missing or corrupt."""
    path = _index_path(cache_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _LOG.info("context index missing or corrupt")
        return None
    if not isinstance(raw, dict):
        return None
    records = _parse_file_records(raw.get(_ENVELOPE_FILES))
    parsed = _parse_chunks(raw.get(_ENVELOPE_CHUNKS))
    if records is None or parsed is None:
        return None
    return records, parsed


def _parse_file_records(raw: Any) -> list[dict[str, str | int]] | None:
    if not isinstance(raw, list):
        return None
    records: list[dict[str, str | int]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        size = item.get("size")
        mtime_ns = item.get("mtime_ns")
        digest = item.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(size, int)
            or not isinstance(mtime_ns, int)
            or not isinstance(digest, str)
            or not digest
        ):
            return None
        records.append(
            {
                "name": name,
                "size": size,
                "mtime_ns": mtime_ns,
                "sha256": digest,
            }
        )
    return records


def _parse_chunks(raw: Any) -> list[dict[str, str]] | None:
    if not isinstance(raw, list):
        return None
    chunks: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        source_file = item.get("source_file")
        chunk_id = item.get("chunk_id")
        text = item.get("text")
        if (
            not isinstance(source_file, str)
            or not isinstance(chunk_id, str)
            or not isinstance(text, str)
        ):
            return None
        chunks.append(
            {
                "source_file": source_file,
                "chunk_id": chunk_id,
                "text": text,
            }
        )
    return chunks


def _write_persisted_index(
    cache_dir: Path,
    records: Sequence[Mapping[str, str | int]],
    chunks: Sequence[Mapping[str, str]],
) -> Path:
    """Atomically replace the persisted index. Does not log chunk text."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _index_path(cache_dir)
    payload = {
        _ENVELOPE_FILES: [dict(record) for record in records],
        _ENVELOPE_CHUNKS: [dict(chunk) for chunk in chunks],
    }
    tmp_path = path.with_suffix(path.suffix + _INDEX_TMP_SUFFIX)
    tmp_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path
