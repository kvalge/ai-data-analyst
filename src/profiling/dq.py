# dq.py

"""Data-quality checks on a sample frame. Not tools until 2.11."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

_LOG = logging.getLogger(__name__)

NULLS_RESULT_KEYS = frozenset({"null_counts", "null_pcts", "row_count"})
DUPLICATES_RESULT_KEYS = frozenset({"duplicate_row_count", "row_count"})
TYPE_MISMATCHES_RESULT_KEYS = frozenset({"type_mismatches", "row_count"})
FORMATTING_RESULT_KEYS = frozenset({"formatting_issues", "row_count"})

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\S*)?$")
_SLASH_DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
_DOT_DATE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}$")
# TODO: US-style grouping (1,234.56) is an arbitrary, unverified assumption;
# European 1.234,56 is not flagged. Revisit locale detection, not a second regex.
_THOUSANDS = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")


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


def detect_type_mismatches(frame: pd.DataFrame) -> dict[str, Any]:
    """Return per-column type issues on `frame`: strings that parse as
    numeric or dates, or mixed parseable and non-parseable values.

    Already-typed numeric/datetime columns are not mismatches. Empty and
    all-null text columns report no issue. Counts come from the given
    frame only. Thousands separators and mixed date formats are 2.6.
    """
    row_count = int(len(frame))
    type_mismatches = {
        str(column): _type_mismatch_kind(series) for column, series in frame.items()
    }
    _LOG.info(
        "detect_type_mismatches row_count=%s mismatches=%s",
        row_count,
        {column: kind for column, kind in type_mismatches.items() if kind is not None},
    )
    return {
        "type_mismatches": type_mismatches,
        "row_count": row_count,
    }


def _type_mismatch_kind(series: pd.Series) -> str | None:
    if not _is_text_series(series):
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    numeric_parsed = pd.Series(pd.to_numeric(non_null, errors="coerce"))
    numeric_ok = int(numeric_parsed.notna().sum())
    if numeric_ok == len(non_null):
        return "numeric_as_string"
    date_parsed = pd.Series(
        pd.to_datetime(non_null, errors="coerce", format="mixed")
    )
    date_ok = int(date_parsed.notna().sum())
    if date_ok == len(non_null):
        return "date_as_string"
    if numeric_ok > 0 or date_ok > 0:
        return "mixed"
    return None


def detect_inconsistent_formatting(frame: pd.DataFrame) -> dict[str, Any]:
    """Return per-column formatting issues on `frame`.

    v1 kinds, in this order when several apply: mixed date formats,
    thousands separators, leading/trailing whitespace. Already-typed
    numeric/datetime columns are skipped. Empty and all-null text
    columns report no issues. Counts come from the given frame only.
    """
    row_count = int(len(frame))
    formatting_issues = {
        str(column): _formatting_issue_kinds(series)
        for column, series in frame.items()
    }
    _LOG.info(
        "detect_inconsistent_formatting row_count=%s issues=%s",
        row_count,
        {
            column: kinds
            for column, kinds in formatting_issues.items()
            if kinds
        },
    )
    return {
        "formatting_issues": formatting_issues,
        "row_count": row_count,
    }


def _formatting_issue_kinds(series: pd.Series) -> list[str]:
    if not _is_text_series(series):
        return []
    texts = [str(value) for value in series.dropna()]
    if not texts:
        return []
    kinds: list[str] = []
    families = {
        family for text in texts if (family := _date_family(text)) is not None
    }
    if len(families) > 1:
        kinds.append("mixed_date_formats")
    if any(_THOUSANDS.fullmatch(text.strip()) for text in texts):
        kinds.append("thousands_separators")
    if any(text != text.strip() for text in texts):
        kinds.append("whitespace")
    return kinds


def _date_family(text: str) -> str | None:
    stripped = text.strip()
    if _ISO_DATE.fullmatch(stripped):
        return "iso"
    if _SLASH_DATE.fullmatch(stripped):
        return "slash"
    if _DOT_DATE.fullmatch(stripped):
        return "dot"
    return None


def _is_text_series(series: pd.Series) -> bool:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return False
    if pd.api.types.is_numeric_dtype(dtype):
        return False
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return False
    return True
