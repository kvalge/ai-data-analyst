"""Validate uploaded analysis data files before they are stored or read."""

from __future__ import annotations

import os
from pathlib import Path

DATA_FILE_SUFFIXES = frozenset({".csv", ".xlsx", ".xls", ".json"})


class FileValidationError(ValueError):
    """The file is not an allowed analysis data upload."""


def validate_data_file(path: Path, *, max_bytes: int) -> Path:
    """Accept a readable allowlisted data file within `max_bytes`.

    Checks suffix, that the path is a non-empty readable file, and size.
    Does not parse or load the contents.
    """
    resolved = path.expanduser()
    if not resolved.is_file():
        raise FileValidationError(
            f"Data file does not exist or is not a file: {path}"
        )
    if not os.access(resolved, os.R_OK):
        raise FileValidationError(f"Data file is not readable: {path}")

    suffix = resolved.suffix.lower()
    if suffix not in DATA_FILE_SUFFIXES:
        allowed = ", ".join(sorted(DATA_FILE_SUFFIXES))
        raise FileValidationError(
            f"Data file suffix {resolved.suffix!r} is not allowed. "
            f"Use one of: {allowed}."
        )

    size = resolved.stat().st_size
    if size == 0:
        raise FileValidationError(f"Data file is empty: {path}")
    if size > max_bytes:
        raise FileValidationError(
            f"Data file is {size} bytes, over the limit of {max_bytes} bytes: {path}"
        )
    return resolved
