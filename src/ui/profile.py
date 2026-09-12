# profile.py

"""Render a profile_source result in Streamlit. Never shows dataset rows."""

from __future__ import annotations

from typing import Any, Iterable

import streamlit as st

from src.ui.profile_step import (
    ACTION_ABORT,
    ACTION_CONTINUE,
    ACTION_SKIP_REMAINING,
    PROFILE_SECTIONS,
    SECTION_DQ,
    SECTION_EDA,
    SECTION_SCHEMA,
)

PROFILE_SOURCE_ID_KEY = "profile_source_id"
SELECTED_SOURCE_ID_KEY = "selected_source_id"


def schema_rows(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per column: dtype and sample null count."""
    dtypes = schema["dtypes"]
    null_counts = schema["null_counts"]
    return [
        {
            "column": column,
            "dtype": dtypes[column],
            "null_count": null_counts[column],
        }
        for column in schema["columns"]
    ]


def null_rows(nulls: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per column: null count and percent."""
    counts = nulls["null_counts"]
    pcts = nulls["null_pcts"]
    return [
        {
            "column": column,
            "null_count": counts[column],
            "null_pct": pcts[column],
        }
        for column in counts
    ]


def mismatch_rows(type_mismatches: dict[str, Any]) -> list[dict[str, Any]]:
    """Columns that have a type-mismatch kind."""
    return [
        {"column": column, "kind": kind}
        for column, kind in type_mismatches.items()
        if kind
    ]


def formatting_rows(formatting_issues: dict[str, Any]) -> list[dict[str, Any]]:
    """Columns that have at least one formatting kind."""
    return [
        {"column": column, "kinds": ", ".join(kinds)}
        for column, kinds in formatting_issues.items()
        if kinds
    ]


def outlier_rows(outliers: dict[str, Any]) -> list[dict[str, Any]]:
    """Numeric columns with Tukey fences (None reports are skipped)."""
    rows: list[dict[str, Any]] = []
    for column, report in outliers.items():
        if report is None:
            continue
        rows.append(
            {
                "column": column,
                "count": report["count"],
                "lower_bound": report["lower_bound"],
                "upper_bound": report["upper_bound"],
            }
        )
    return rows


def numeric_summary_rows(numeric: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per numeric column with moments."""
    return [{"column": column, **stats} for column, stats in numeric.items()]


def categorical_summary_rows(categorical: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per categorical column with nunique."""
    return [
        {"column": column, "nunique": report["nunique"]}
        for column, report in categorical.items()
    ]


def histogram_rows(hist: dict[str, Any]) -> list[dict[str, Any]]:
    """Bin low/high/count rows for a numeric histogram."""
    counts = hist["counts"]
    edges = hist["edges"]
    return [
        {
            "bin": index,
            "low": edges[index],
            "high": edges[index + 1],
            "count": counts[index],
        }
        for index in range(len(counts))
    ]


def categorical_top_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Top-N value counts, plus other_count when it is nonzero."""
    rows = [
        {"value": item["value"], "count": item["count"]} for item in report["top"]
    ]
    other_count = report["other_count"]
    if other_count:
        rows.append({"value": "(other)", "count": other_count})
    return rows


def pearson_rows(corr: dict[str, Any]) -> list[dict[str, Any]]:
    """Square Pearson matrix as row dicts, or empty when skipped."""
    if corr["skipped"] or corr["pearson"] is None:
        return []
    columns = corr["columns"]
    pearson = corr["pearson"]
    return [
        {"column": row, **{column: pearson[row][column] for column in columns}}
        for row in columns
    ]


def render_closable_heading(title: str, *, close_key: str) -> bool:
    """Render a panel heading with Close. True if Close was clicked."""
    title_col, close_col = st.columns([6, 1])
    with title_col:
        st.subheader(title)
    with close_col:
        return st.button("Close", key=close_key)


def render_profile(
    result: dict[str, Any],
    *,
    sections: Iterable[str] = PROFILE_SECTIONS,
) -> None:
    """Show selected schema / DQ / EDA sections from a profile_source result."""
    shown = tuple(sections)
    cached = "cached" if result["cached"] else "fresh"
    file_row_count = result["schema"].get("file_row_count")
    file_bit = (
        f"file rows {file_row_count}"
        if file_row_count is not None
        else "file row count unknown"
    )
    st.caption(
        f"{cached} · {result['sample_row_count']} profiled row(s) · {file_bit}. "
        "Quick overview of a bounded head, not a whole-file profile."
    )
    if SECTION_SCHEMA in shown:
        render_schema_section(result["schema"])
    if SECTION_DQ in shown:
        render_dq_section(result["dq"])
    if SECTION_EDA in shown:
        render_eda_section(result["eda"])


def render_guided_stepper() -> str | None:
    """Continue, Skip remaining, or Abort. None if no button was clicked."""
    continue_col, skip_col, abort_col = st.columns(3)
    with continue_col:
        if st.button("Continue", key="profile_continue"):
            return ACTION_CONTINUE
    with skip_col:
        if st.button("Skip remaining", key="profile_skip_remaining"):
            return ACTION_SKIP_REMAINING
    with abort_col:
        if st.button("Abort", key="profile_abort"):
            return ACTION_ABORT
    return None


def render_schema_section(schema: dict[str, Any]) -> None:
    """Render the schema table."""
    st.markdown("#### Schema")
    rows = schema_rows(schema)
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.caption("No columns.")


def render_dq_section(dq: dict[str, Any]) -> None:
    """Render nulls, duplicates, type mismatches, formatting, and outliers."""
    st.markdown("#### Data quality")
    st.caption(f"Duplicates (extra full rows): {dq['duplicates']['duplicate_row_count']}")
    _table_or_caption("Nulls", null_rows(dq["nulls"]))
    _table_or_caption("Type mismatches", mismatch_rows(dq["type_mismatches"]["type_mismatches"]))
    _table_or_caption(
        "Formatting", formatting_rows(dq["formatting"]["formatting_issues"])
    )
    _table_or_caption("Outliers", outlier_rows(dq["outliers"]["outliers"]))


def render_eda_section(eda: dict[str, Any]) -> None:
    """Render summary stats, distributions, and correlations."""
    st.markdown("#### EDA")
    summary = eda["summary"]
    _table_or_caption("Numeric summary", numeric_summary_rows(summary["numeric"]))
    _table_or_caption(
        "Categorical summary", categorical_summary_rows(summary["categorical"])
    )
    st.markdown("##### Distributions")
    for column, hist in eda["distributions"]["numeric"].items():
        with st.expander(f"Histogram: {column}", expanded=False):
            rows = histogram_rows(hist)
            if rows:
                st.dataframe(rows, hide_index=True)
            else:
                st.caption("No bins.")
    for column, report in eda["distributions"]["categorical"].items():
        with st.expander(f"Top values: {column}", expanded=False):
            rows = categorical_top_rows(report)
            if rows:
                st.dataframe(rows, hide_index=True)
            else:
                st.caption("No values.")
    corr = eda["correlations"]
    st.markdown("##### Correlations")
    if corr["skipped"]:
        st.caption("Pearson skipped (fewer than two numeric columns).")
    else:
        rows = pearson_rows(corr)
        if rows:
            st.dataframe(rows, hide_index=True)


def _table_or_caption(title: str, rows: list[dict[str, Any]]) -> None:
    st.markdown(f"##### {title}")
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.caption("None found.")
