# code_review.py

"""Streamlit widgets for the approve_code interrupt. Does not run the code."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.agent.code_approval import ACTION_APPROVE, ACTION_REJECT


def render_code_review(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Show generated SQL or Python with Approve / Reject. No textarea yet."""
    st.info("Approve or reject this generated SQL or Python. Rejected code is not run.")
    source_id = str(payload.get("source_id") or "")
    if source_id:
        st.caption(f"Source: {source_id}")
    rationale = str(payload.get("rationale") or "").strip()
    if rationale:
        st.caption(rationale)
    st.code(str(payload.get("code") or ""), language=None)
    approve_col, reject_col = st.columns(2)
    if approve_col.button("Approve", key="code_graph_approve"):
        return {"action": ACTION_APPROVE}
    if reject_col.button("Reject", key="code_graph_reject"):
        return {"action": ACTION_REJECT}
    return None
