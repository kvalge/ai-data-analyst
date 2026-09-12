# chat.py

"""Streamlit chat that sends user text into the graph. No system prompts here."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import streamlit as st
from langgraph.types import Command

from src.agent.code_approval import KIND_APPROVE_CODE
from src.agent.confirm_sources import KIND_CONFIRM_SOURCES
from src.agent.graph import build_graph
from src.agent.profile_steps import KIND_PROFILE_STEP
from src.agent.state import AgentMessage, empty_agent_state
from src.config import Settings
from src.ui.code_review import render_code_review
from src.ui.hitl import HITL_MODE_KEY, resolve_hitl_mode
from src.ui.profile_pause import render_profile_pause
from src.ui.source_confirm import render_source_confirm

_INTERRUPT_KINDS = frozenset(
    {KIND_CONFIRM_SOURCES, KIND_PROFILE_STEP, KIND_APPROVE_CODE}
)

_LOG = logging.getLogger(__name__)

GRAPH_KEY = "agent_graph"
THREAD_ID_KEY = "agent_thread_id"
CHAT_ERROR_KEY = "chat_error"


def ensure_thread_id(session_state: Any) -> str:
    """Create one graph thread id for this Streamlit session."""
    thread_id = session_state.get(THREAD_ID_KEY)
    if not isinstance(thread_id, str) or not thread_id.strip():
        thread_id = str(uuid.uuid4())
        session_state[THREAD_ID_KEY] = thread_id
    return thread_id


def ensure_graph(session_state: Any, settings: Settings) -> Any:
    """Reuse the compiled graph so MemorySaver keeps the thread."""
    graph = session_state.get(GRAPH_KEY)
    if graph is None:
        graph = build_graph(settings=settings)
        session_state[GRAPH_KEY] = graph
    return graph


def thread_config(thread_id: str) -> dict[str, Any]:
    """LangGraph checkpointer config for one conversation thread."""
    return {"configurable": {"thread_id": thread_id}}


def graph_messages(graph: Any, thread_id: str) -> list[AgentMessage]:
    """Return checkpointed user/assistant messages. Not the system prompt."""
    snap = graph.get_state(thread_config(thread_id))
    values = snap.values or {}
    messages = values.get("messages") or []
    return list(messages)


def graph_error(graph: Any, thread_id: str) -> str | None:
    """Return the last visible graph error, if any."""
    snap = graph.get_state(thread_config(thread_id))
    values = snap.values or {}
    error = values.get("error")
    if isinstance(error, str) and error.strip():
        return error
    return None


def store_invoke_result(session_state: Any, result: dict[str, Any]) -> None:
    """Keep the invoke error across rerun when the checkpoint was never written."""
    error = result.get("error")
    if isinstance(error, str) and error.strip():
        session_state[CHAT_ERROR_KEY] = error
        return
    session_state.pop(CHAT_ERROR_KEY, None)


def visible_chat_error(
    checkpoint_error: str | None, session_state: Any
) -> str | None:
    """Show the invoke/session error even when the checkpoint has nothing."""
    stored = session_state.get(CHAT_ERROR_KEY)
    if isinstance(stored, str) and stored.strip():
        return stored
    if isinstance(checkpoint_error, str) and checkpoint_error.strip():
        return checkpoint_error
    return None


def graph_interrupt_payload(graph: Any, thread_id: str) -> dict[str, Any] | None:
    """Return a known interrupt value, if the graph is paused."""
    snap = graph.get_state(thread_config(thread_id))
    interrupts = getattr(snap, "interrupts", ()) or ()
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    if isinstance(value, dict) and value.get("kind") in _INTERRUPT_KINDS:
        return value
    return None


def graph_profile_summary(graph: Any, thread_id: str) -> dict[str, Any] | None:
    """Return the checkpointed compact profile, if one was written."""
    snap = graph.get_state(thread_config(thread_id))
    values = snap.values or {}
    summary = values.get("profile_summary")
    if isinstance(summary, dict):
        return summary
    return None


def resume_interrupt(
    graph: Any,
    *,
    decision: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    """Resume a graph interrupt with the user's decision."""
    return _invoke_or_error(
        graph,
        Command(resume=decision),
        thread_config(thread_id),
        thread_id,
    )


def invoke_user_turn(
    graph: Any,
    *,
    user_text: str,
    hitl_mode: str,
    thread_id: str,
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Send one user line into the graph. Does not attach a system prompt."""
    text = user_text.strip()
    if not text:
        raise ValueError("Chat text is empty.")
    chosen = [item for item in (source_ids or []) if item]
    config = thread_config(thread_id)
    _LOG.info("UI chat turn")
    if graph_messages(graph, thread_id):
        return _invoke_or_error(
            graph,
            {
                "messages": [{"role": "user", "content": text}],
                "hitl_mode": hitl_mode,
                "source_ids": chosen,
            },
            config,
            thread_id,
        )
    state = empty_agent_state(hitl_mode=hitl_mode)
    state["messages"] = [{"role": "user", "content": text}]
    state["source_ids"] = chosen
    return _invoke_or_error(graph, state, config, thread_id)


def _invoke_or_error(
    graph: Any,
    payload: Any,
    config: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    """Run invoke. Escape exceptions so Streamlit does not crash the page."""
    try:
        return graph.invoke(payload, config)
    except Exception as exc:
        # Parse retries live in the graph. Escape leftover invoke crashes here.
        _LOG.info("UI chat invoke failed")
        message = str(exc).strip() or "Chat failed."
        return {"error": message, "messages": graph_messages(graph, thread_id)}


def render_chat(
    settings: Settings, *, source_ids: list[str] | None = None
) -> None:
    """Show the transcript and send typed text into the graph."""
    graph = ensure_graph(st.session_state, settings)
    thread_id = ensure_thread_id(st.session_state)
    for message in graph_messages(graph, thread_id):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        with st.chat_message(role):
            st.write(message.get("content", ""))
    error = visible_chat_error(graph_error(graph, thread_id), st.session_state)
    if error:
        st.error(error)
    pending = graph_interrupt_payload(graph, thread_id)
    if pending is not None:
        kind = pending.get("kind")
        decision: dict[str, Any] | None = None
        caption = "Confirm, select, or abort the data source before chatting."
        if kind == KIND_CONFIRM_SOURCES:
            decision = render_source_confirm(pending)
        elif kind == KIND_PROFILE_STEP:
            decision = render_profile_pause(
                pending, graph_profile_summary(graph, thread_id)
            )
            caption = (
                "Continue, skip remaining, or abort profiling before chatting."
            )
        elif kind == KIND_APPROVE_CODE:
            decision = render_code_review(pending)
            caption = "Approve or reject generated SQL or Python before chatting."
        if decision is not None:
            result = resume_interrupt(
                graph, decision=decision, thread_id=thread_id
            )
            store_invoke_result(st.session_state, result)
            st.rerun()
        st.caption(caption)
        return
    if not settings.ollama_model_primary:
        st.caption("Set OLLAMA_MODEL_* in .env to enable chat.")
    typed = st.chat_input("Ask about your data")
    if typed is None or not typed.strip():
        return
    if not settings.ollama_model_primary:
        st.error("Set OLLAMA_MODEL_PRIMARY in .env before chatting.")
        return
    mode = resolve_hitl_mode(st.session_state.get(HITL_MODE_KEY))
    try:
        with st.spinner("Waiting for the local model…"):
            result = invoke_user_turn(
                graph,
                user_text=typed,
                hitl_mode=str(mode),
                thread_id=thread_id,
                source_ids=source_ids,
            )
    except Exception as exc:
        st.session_state[CHAT_ERROR_KEY] = str(exc).strip() or "Chat failed."
        st.rerun()
        return
    store_invoke_result(st.session_state, result)
    st.rerun()
