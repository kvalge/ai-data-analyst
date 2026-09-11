# eda.py

"""EDA summaries for a pandas DataFrame.

The caller decides whether `frame` is a sample or a full load; this
module does not sample or cap rows. Not a tool until 2.11.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

SUMMARY_STATS_RESULT_KEYS = frozenset({"numeric", "categorical", "row_count"})
DISTRIBUTIONS_RESULT_KEYS = frozenset({"numeric", "categorical", "row_count"})
CORRELATIONS_RESULT_KEYS = frozenset({"columns", "pearson", "skipped", "row_count"})

_HIST_BINS = 10
_TOP_N = 10


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


def distributions(frame: pd.DataFrame) -> dict[str, Any]:
    """Return numeric histogram bins and categorical top-N value counts.

    Bool is treated as categorical, not numeric. Empty numeric columns
    report empty counts/edges. Categories beyond top-N are summed in
    `other_count`. Results describe only the rows in `frame` (sample or
    full). Compact JSON only; no plots.
    """
    row_count = int(len(frame))
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    for column, series in frame.items():
        name = str(column)
        if _is_numeric_column(series):
            numeric[name] = _numeric_histogram(series)
        else:
            categorical[name] = _categorical_top(series)
    _LOG.info(
        "distributions row_count=%s numeric=%s categorical=%s",
        row_count,
        list(numeric),
        list(categorical),
    )
    return {
        "numeric": numeric,
        "categorical": categorical,
        "row_count": row_count,
    }


def correlations(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a Pearson matrix for numeric columns, or skip if fewer than two.

    Bool is not numeric. `pearson` is None when skipped. Pairwise NaN
    (constant column, no overlap) becomes JSON null. Results describe
    only the rows in `frame` (sample or full).
    """
    row_count = int(len(frame))
    numeric_items = [
        (str(name), series)
        for name, series in frame.items()
        if _is_numeric_column(series)
    ]
    columns = [name for name, _ in numeric_items]
    if len(columns) < 2:
        _LOG.info(
            "correlations skipped columns=%s row_count=%s",
            columns,
            row_count,
        )
        return {
            "columns": columns,
            "pearson": None,
            "skipped": True,
            "row_count": row_count,
        }
    subset = pd.DataFrame(dict(numeric_items))
    corr = subset.corr(method="pearson")
    pearson = {
        str(row): {
            str(column): _corr_cell(corr.loc[row, column])
            for column in corr.columns
        }
        for row in corr.index
    }
    _LOG.info("correlations columns=%s row_count=%s", columns, row_count)
    return {
        "columns": columns,
        "pearson": pearson,
        "skipped": False,
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


def _numeric_histogram(series: pd.Series) -> dict[str, Any]:
    numeric = series.dropna().astype("float64")
    if numeric.empty:
        return {"counts": [], "edges": []}
    hist_counts, hist_edges = np.histogram(numeric.to_numpy(), bins=_HIST_BINS)
    return {
        "counts": [int(count) for count in hist_counts],
        "edges": [float(edge) for edge in hist_edges],
    }


def _categorical_top(series: pd.Series) -> dict[str, Any]:
    counts = series.dropna().value_counts()
    top = counts.head(_TOP_N)
    other_count = int(counts.iloc[_TOP_N:].sum()) if len(counts) > _TOP_N else 0
    return {
        "top": [
            {"value": _json_cell(value), "count": int(count)}
            for value, count in top.items()
        ],
        "other_count": other_count,
    }


def _json_cell(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)):
        return value
    return str(value)


def _corr_cell(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _is_numeric_column(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series.dtype):
        return False
    return bool(pd.api.types.is_numeric_dtype(series.dtype))
