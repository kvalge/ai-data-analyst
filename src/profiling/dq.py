# dq.py

"""Data-quality checks on a sample frame. Not tools until 2.11."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_LOG = logging.getLogger(__name__)

NULLS_RESULT_KEYS = frozenset({"null_counts", "null_pcts", "row_count"})
DUPLICATES_RESULT_KEYS = frozenset({"duplicate_row_count", "row_count"})


def detect_nulls(frame: pd.DataFrame) -> dict[str, Any]:
    """Return per-column null counts and percentages for `frame`.

    Percentages are 0–100 relative to `len(frame)`. An empty frame reports
    0.0% for every column. Counts come from the given frame only.
    """
    row_count = int(len(frame))
    na_counts = frame.isna().sum().to_dict()
    null_counts = {str(column): int(count) for column, count in na_counts.items()}
    null_pcts = {
        column: (100.0 * count / row_count) if row_count else 0.0
        for column, count in null_counts.items()
    }
    _LOG.info(
        "detect_nulls columns=%s row_count=%s",
        list(null_counts),
        row_count,
    )
    return {
        "null_counts": null_counts,
        "null_pcts": null_pcts,
        "row_count": row_count,
    }


def detect_duplicates(frame: pd.DataFrame) -> dict[str, Any]:
    """Return how many extra full-row copies `frame` contains.

    v1 compares every column. `duplicate_row_count` is rows you would drop
    with `drop_duplicates` (keep first). An empty frame reports 0.
    """
    row_count = int(len(frame))
    duplicate_row_count = int(frame.duplicated(keep="first").sum())
    _LOG.info(
        "detect_duplicates row_count=%s duplicate_row_count=%s",
        row_count,
        duplicate_row_count,
    )
    return {
        "duplicate_row_count": duplicate_row_count,
        "row_count": row_count,
    }
