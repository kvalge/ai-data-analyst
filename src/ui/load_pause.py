# load_pause.py

"""Streamlit widgets for the approve_load interrupt. Does not load the file."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.agent.load_approval import ACTION_APPROVE, ACTION_REJECT


def render_load_pause(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Show over-limit metadata with Approve / Reject. No dataset rows."""
    st.info(
        "This full-file load is over the configured limit. "
        "Approve to load anyway, or reject. Rejected loads do not run."
    )
    source_id = str(payload.get("source_id") or "")
    if source_id:
        st.caption(f"Source: {source_id}")
    reason = str(payload.get("reason") or "").strip()
    if reason:
        st.caption(f"Reason: {reason}")
    st.caption(
        f"Rows: {payload.get('row_count')} "
        f"(limit {payload.get('max_full_load_rows')})"
    )
    st.caption(
        f"Bytes: {payload.get('size_bytes')} "
        f"(limit {payload.get('max_bytes')})"
    )
    approve_col, reject_col = st.columns(2)
    if approve_col.button("Approve", key="load_graph_approve"):
        return {"action": ACTION_APPROVE}
    if reject_col.button("Reject", key="load_graph_reject"):
        return {"action": ACTION_REJECT}
    return None
