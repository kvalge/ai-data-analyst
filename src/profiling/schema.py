# schema.py

"""Detect a stable schema dict from a sample frame. Does not return row data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

_LOG = logging.getLogger(__name__)

SCHEMA_RESULT_KEYS = frozenset(
    {
        "columns",
        "dtypes",
        "null_counts",
        "sample_row_count",
        "file_row_count",
    }
)


def detect_schema(
    frame: pd.DataFrame,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Return columns, dtypes, and sample null counts.

    `file_row_count` is the full CSV line count when `path` is a `.csv`;
    otherwise it is None (unknown). Null counts come from `frame` only.
    """
    columns = [str(column) for column in frame.columns]
    dtypes = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    na_counts = frame.isna().sum().to_dict()
    null_counts = {str(column): int(na_counts[column]) for column in frame.columns}
    file_row_count = cheap_file_row_count(path) if path is not None else None
    _LOG.info(
        "detect_schema columns=%s sample_rows=%s file_row_count=%s path=%s",
        columns,
        len(frame),
        file_row_count,
        path,
    )
    return {
        "columns": columns,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "sample_row_count": int(len(frame)),
        "file_row_count": file_row_count,
    }


def cheap_file_row_count(path: Path) -> int | None:
    """Count CSV data rows (after the header) by scanning lines. Else unknown."""
    if path.suffix.lower() != ".csv":
        return None
    # Line scan, not a CSV parse: quoted fields with embedded newlines overcount.
    with path.open(encoding="utf-8", newline="") as handle:
        header = handle.readline()
        if not header:
            return 0
        return sum(1 for _ in handle)
