# eda.py

"""EDA summaries for a pandas DataFrame.

The caller decides whether `frame` is a sample or a full load; this
module does not sample or cap rows. Not a tool until 2.11.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_LOG = logging.getLogger(__name__)

SUMMARY_STATS_RESULT_KEYS = frozenset({"numeric", "categorical", "row_count"})


def summary_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Return numeric count/mean/median/min/max and categorical nunique.

    Bool is treated as categorical, not numeric. Empty numeric columns
    report count 0 and null mean/median/min/max. Results describe only
    the rows in `frame` (sample or full).
    """
    row_count = int(len(frame))
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    for column, series in frame.items():
        name = str(column)
        if _is_numeric_column(series):
            numeric[name] = _numeric_summary(series)
        else:
            categorical[name] = {"nunique": int(series.nunique(dropna=True))}
    _LOG.info(
        "summary_stats row_count=%s numeric=%s categorical=%s",
        row_count,
        list(numeric),
        {column: report["nunique"] for column, report in categorical.items()},
    )
    return {
        "numeric": numeric,
        "categorical": categorical,
        "row_count": row_count,
    }


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = series.dropna().astype("float64")
    if numeric.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def _is_numeric_column(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series.dtype):
        return False
    return bool(pd.api.types.is_numeric_dtype(series.dtype))
