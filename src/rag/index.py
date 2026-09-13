# index.py

"""Build an in-memory context index from CONTEXT_DIR. No persist yet."""

from __future__ import annotations

import logging
from pathlib import Path

from src.rag.chunker import chunk_context
from src.rag.readers import read_context_file
from src.rag.retrieve import DomainContextIndex
from src.storage.context import list_context_files

_LOG = logging.getLogger(__name__)


def build_context_index(context_dir: Path, *, max_bytes: int) -> DomainContextIndex:
    """Read and chunk allowlisted files in `context_dir`. Never data files."""
    chunks: list[dict[str, str]] = []
    files = list_context_files(context_dir)
    for path in files:
        document = read_context_file(path, max_bytes=max_bytes)
        chunks.extend(chunk_context(document))
    _LOG.info("build context index files=%s chunks=%s", len(files), len(chunks))
    return DomainContextIndex(chunks)
