# code_review.py

"""Streamlit widgets for the approve_code interrupt. Does not run the code."""

from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st

from src.agent.code_approval import (
    ACTION_APPROVE,
    ACTION_EDIT_RUN,
    ACTION_REJECT,
    TOOL_QUERY_DATABASE,
)

_SEEN_KEY = "code_graph_payload"
_KEYS_KEY = "code_graph_keys"


def code_review_decision(
    *,
    action: str,
    original: str,
    edited: str,
) -> dict[str, Any]:
    """Map Approve/Reject plus the textarea to a resume value.

    Does not execute SQL or Python. Unchanged Approve keeps the original
    tool arguments; a changed Approve is edit_run.
    """
    if action != ACTION_APPROVE:
        return {"action": action}
    if edited == original:
        return {"action": ACTION_APPROVE}
    return {"action": ACTION_EDIT_RUN, "code": edited}


def code_review_text_identity(payload: dict[str, Any]) -> tuple[str, str, str]:
    """Tool, source, and generated text that identify one approve_code pause."""
    return (
        str(payload.get("tool") or ""),
        str(payload.get("source_id") or ""),
        str(payload.get("code") or ""),
    )


def code_review_widget_keys(payload: dict[str, Any]) -> dict[str, str]:
    """Streamlit keys for this interrupt. A new payload remounts the widgets."""
    material = "\0".join(code_review_text_identity(payload))
    suffix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return {
        "text": f"code_graph_text_{suffix}",
        "approve": f"code_graph_approve_{suffix}",
        "reject": f"code_graph_reject_{suffix}",
    }


def sync_code_review_session(
    session: Any, payload: dict[str, Any]
) -> dict[str, str]:
    """Drop a leftover textarea when the interrupt payload changes.

    `session` is Streamlit's proxy or a plain dict in tests.
    """
    keys = code_review_widget_keys(payload)
    identity = code_review_text_identity(payload)
    if session.get(_SEEN_KEY) == identity:
        return keys
    previous = session.get(_KEYS_KEY)
    if isinstance(previous, dict):
        text_key = previous.get("text")
        if isinstance(text_key, str):
            session.pop(text_key, None)
    session[_SEEN_KEY] = identity
    session[_KEYS_KEY] = keys
    session.pop(keys["text"], None)
    return keys


def forget_code_review_session(session: Any, keys: dict[str, str]) -> None:
    """Clear widget state after a decision so the next pause is not stale."""
    session.pop(keys["text"], None)
    session.pop(_SEEN_KEY, None)
    session.pop(_KEYS_KEY, None)


def render_code_review(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Show generated SQL or Python in an editable textarea with Approve / Reject."""
    keys = sync_code_review_session(st.session_state, payload)
    st.info(
        "Edit if needed, then approve or reject this generated SQL or Python. "
        "Rejected or empty edits are not run. This page does not execute the code."
    )
    source_id = str(payload.get("source_id") or "")
    if source_id:
        st.caption(f"Source: {source_id}")
    rationale = str(payload.get("rationale") or "").strip()
    if rationale:
        st.caption(rationale)
    original = str(payload.get("code") or "")
    label = "SQL" if payload.get("tool") == TOOL_QUERY_DATABASE else "Python"
    edited = st.text_area(
        label,
        value=original,
        height=240,
        key=keys["text"],
    )
    approve_col, reject_col = st.columns(2)
    if approve_col.button("Approve", key=keys["approve"]):
        decision = code_review_decision(
            action=ACTION_APPROVE,
            original=original,
            edited=str(edited),
        )
        forget_code_review_session(st.session_state, keys)
        return decision
    if reject_col.button("Reject", key=keys["reject"]):
        decision = code_review_decision(
            action=ACTION_REJECT,
            original=original,
            edited=str(edited),
        )
        forget_code_review_session(st.session_state, keys)
        return decision
    return None
