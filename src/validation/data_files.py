# data_files.py

"""Validate uploaded analysis data files before they are stored or read."""

from __future__ import annotations

from pathlib import Path

from src.validation.uploads import FileValidationError, validate_upload

DATA_FILE_SUFFIXES = frozenset({".csv", ".xlsx", ".xls", ".json"})

__all__ = ["DATA_FILE_SUFFIXES", "FileValidationError", "validate_data_file"]


def validate_data_file(path: Path, *, max_bytes: int) -> Path:
    """Accept a readable allowlisted data file within `max_bytes`.

    Checks suffix, that the path is a non-empty readable file, and size.
    Does not parse or load the contents.
    """
    return validate_upload(
        path,
        max_bytes=max_bytes,
        suffixes=DATA_FILE_SUFFIXES,
        kind="Data file",
    )
