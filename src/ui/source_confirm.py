# source_confirm.py

"""Streamlit widgets for the confirm_sources interrupt. Does not run the graph."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.agent.confirm_sources import (
    ACTION_ABORT,
    ACTION_CONFIRM,
    ACTION_SELECT,
    REASON_EMPTY_SCHEMA,
    REASON_MULTIPLE_SOURCES,
    REASON_NO_SOURCES,
)

_REASON_TEXT = {
    REASON_NO_SOURCES: "No data sources are registered. Upload a file, then confirm.",
    REASON_MULTIPLE_SOURCES: "More than one source is available. Select which to use.",
    REASON_EMPTY_SCHEMA: "The selected source has an empty sample or schema.",
}


def reason_message(reason: str) -> str:
    """User-facing text for an interrupt reason. No dataset rows."""
    return _REASON_TEXT.get(reason, "Confirm the data source before continuing.")


def render_source_confirm(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Show confirm / select / abort. Return a decision when a button is clicked."""
    reason = str(payload.get("reason") or "")
    sources = payload.get("sources") or []
    st.info(reason_message(reason))
    if not isinstance(sources, list):
        sources = []
    labels: dict[str, str] = {}
    options: list[str] = []
    for row in sources:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id") or "")
        if not source_id:
            continue
        name = str(row.get("original_name") or source_id)
        kind = str(row.get("kind") or "")
        labels[source_id] = f"{name} ({kind})" if kind else name
        options.append(source_id)
        st.write(labels[source_id])
    selected = None
    if options:
        current = payload.get("source_ids") or []
        default = current[0] if current and current[0] in options else options[0]
        selected = st.selectbox(
            "Source to use",
            options=options,
            index=options.index(default),
            format_func=lambda source_id: labels[source_id],
            key="confirm_source_choice",
        )
    col_confirm, col_select, col_abort = st.columns(3)
    if col_confirm.button("Confirm"):
        return {
            "action": ACTION_CONFIRM,
            "source_ids": [selected] if selected else [],
        }
    if col_select.button("Select", disabled=not selected):
        return {"action": ACTION_SELECT, "source_ids": [selected]}
    if col_abort.button("Abort"):
        return {"action": ACTION_ABORT}
    return None
