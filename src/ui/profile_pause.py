# profile_pause.py

"""Streamlit widgets for a Guided profile_step interrupt. Does not run the graph."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.agent.profile_steps import (
    ACTION_ABORT,
    ACTION_CONTINUE,
    ACTION_SKIP_REMAINING,
    SECTION_DQ,
    SECTION_EDA,
    SECTION_SCHEMA,
    visible_profile_sections,
)
from src.ui.profile import render_profile


def profile_result_for_pause(
    payload: dict[str, Any], stored_summary: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Prefer the checkpointed compact summary; else the interrupt section."""
    if isinstance(stored_summary, dict) and stored_summary.get("schema") is not None:
        return stored_summary
    step = str(payload.get("step") or "")
    section = payload.get("section")
    if not isinstance(section, dict):
        return None
    empty_schema = {"columns": [], "dtypes": {}, "null_counts": {}}
    return {
        "cached": bool(payload.get("cached")),
        "sample_row_count": int(payload.get("sample_row_count") or 0),
        "schema": section if step == SECTION_SCHEMA else empty_schema,
        "dq": section if step == SECTION_DQ else {},
        "eda": section if step == SECTION_EDA else {},
    }


def render_profile_pause(
    payload: dict[str, Any], stored_summary: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Show the current profile step and Continue / Skip remaining / Abort."""
    step = str(payload.get("step") or "")
    st.info(
        f"Guided pause after {step}. Continue, skip remaining, or abort."
    )
    result = profile_result_for_pause(payload, stored_summary)
    if result is not None:
        sections = (
            visible_profile_sections(step)
            if isinstance(stored_summary, dict) and stored_summary.get("schema")
            else (step,) if step else visible_profile_sections(SECTION_SCHEMA)
        )
        render_profile(result, sections=sections)
    continue_col, skip_col, abort_col = st.columns(3)
    with continue_col:
        if st.button("Continue", key="profile_graph_continue"):
            return {"action": ACTION_CONTINUE}
    with skip_col:
        if st.button("Skip remaining", key="profile_graph_skip_remaining"):
            return {"action": ACTION_SKIP_REMAINING}
    with abort_col:
        if st.button("Abort", key="profile_graph_abort"):
            return {"action": ACTION_ABORT}
    return None
