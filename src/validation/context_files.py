# context_files.py

"""Validate uploaded business-context documents before they are stored."""

from __future__ import annotations

from pathlib import Path

from src.validation.uploads import FileValidationError, validate_upload

CONTEXT_FILE_SUFFIXES = frozenset({".md", ".txt", ".pdf"})

__all__ = ["CONTEXT_FILE_SUFFIXES", "FileValidationError", "validate_context_file"]


def validate_context_file(path: Path, *, max_bytes: int) -> Path:
    """Accept a readable allowlisted context file within `max_bytes`.

    Checks suffix, that the path is a non-empty readable file, and size.
    Does not parse, index, or execute the contents.
    """
    return validate_upload(
        path,
        max_bytes=max_bytes,
        suffixes=CONTEXT_FILE_SUFFIXES,
        kind="Context file",
    )
