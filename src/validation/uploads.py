# uploads.py

"""Shared suffix, size, and readability checks for uploaded files."""

from __future__ import annotations

import os
from pathlib import Path


class FileValidationError(ValueError):
    """The file is not an allowed upload."""


def validate_upload(
    path: Path,
    *,
    max_bytes: int,
    suffixes: frozenset[str],
    kind: str,
) -> Path:
    """Accept a readable allowlisted file within `max_bytes`.

    Checks suffix, that the path is a non-empty readable file, and size.
    Does not parse or load the contents.
    """
    resolved = path.expanduser()
    if not resolved.is_file():
        raise FileValidationError(f"{kind} does not exist or is not a file: {path}")
    if not os.access(resolved, os.R_OK):
        raise FileValidationError(f"{kind} is not readable: {path}")

    suffix = resolved.suffix.lower()
    if suffix not in suffixes:
        allowed = ", ".join(sorted(suffixes))
        raise FileValidationError(
            f"{kind} suffix {resolved.suffix!r} is not allowed. "
            f"Use one of: {allowed}."
        )

    size = resolved.stat().st_size
    if size == 0:
        raise FileValidationError(f"{kind} is empty: {path}")
    if size > max_bytes:
        raise FileValidationError(
            f"{kind} is {size} bytes, over the limit of {max_bytes} bytes: {path}"
        )
    return resolved
