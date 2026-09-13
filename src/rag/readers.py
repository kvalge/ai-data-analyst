# readers.py

"""Read uploaded context documents as text. No chunking or retrieval here."""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from src.validation.context_files import validate_context_file
from src.validation.uploads import FileValidationError

_LOG = logging.getLogger(__name__)

_PLAIN_SUFFIXES = frozenset({".md", ".txt"})


class ContextReadError(ValueError):
    """A context document could not be read as text."""


def read_context_file(path: Path, *, max_bytes: int) -> dict[str, str]:
    """Return `{source_file, text}` for one allowlisted context document.

    `source_file` is the basename (the document identity). Does not return
    tables or file bytes. `max_bytes` is the same upload cap used to store
    the file.
    """
    resolved = validate_context_file(path, max_bytes=max_bytes)
    suffix = resolved.suffix.lower()
    _LOG.info("read context name=%s suffix=%s", resolved.name, suffix)
    try:
        if suffix in _PLAIN_SUFFIXES:
            text = _read_plain(resolved)
        elif suffix == ".pdf":
            text = _read_pdf(resolved)
        else:
            raise ContextReadError(f"Unsupported context suffix: {suffix}")
    except (ContextReadError, FileValidationError):
        raise
    except (OSError, UnicodeError) as exc:
        _LOG.info("read context failed name=%s", resolved.name)
        raise ContextReadError(
            f"Could not read context file {resolved.name}."
        ) from exc
    if not text.strip():
        raise ContextReadError(
            f"Context file {resolved.name} has no extractable text."
        )
    return {"source_file": resolved.name, "text": text}


def _read_plain(path: Path) -> str:
    """Decode markdown or text as UTF-8. Does not execute the file."""
    return path.read_text(encoding="utf-8-sig")


def _read_pdf(path: Path) -> str:
    """Extract page text from an already-uploaded PDF. Local only."""
    # TODO(8.2): extracted text is unbounded; trim at MAX_PROMPT_CHARS.
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except (OSError, PyPdfError, ValueError) as exc:
        _LOG.info("read context failed name=%s", path.name)
        raise ContextReadError(
            f"Could not read context file {path.name}."
        ) from exc
    return "\n".join(pages)
