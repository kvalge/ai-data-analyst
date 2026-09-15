# chat.py

"""Streamlit chat that sends user text into the graph. No system prompts here."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.types import Command

from src.agent.checkpoint import (
    load_persisted_thread_id,
    persist_thread_id,
    sqlite_checkpointer,
)
from src.agent.code_approval import KIND_APPROVE_CODE
from src.agent.confirm_sources import KIND_CONFIRM_SOURCES
from src.agent.graph import build_graph
from src.agent.load_approval import KIND_APPROVE_LOAD
from src.agent.profile_steps import KIND_PROFILE_STEP
from src.agent.state import AgentMessage, empty_agent_state
from src.config import Settings
from src.ui.code_review import render_code_review
from src.ui.hitl import HITL_MODE_KEY, resolve_hitl_mode
from src.ui.load_pause import render_load_pause
from src.ui.profile_pause import render_profile_pause
from src.ui.artifacts import render_chart_artifacts, render_table_artifacts
from src.ui.report import render_report_download
from src.ui.source_confirm import render_source_confirm

_INTERRUPT_KINDS = frozenset(
    {
        KIND_CONFIRM_SOURCES,
        KIND_PROFILE_STEP,
        KIND_APPROVE_CODE,
        KIND_APPROVE_LOAD,
    }
)

_LOG = logging.getLogger(__name__)

GRAPH_KEY = "agent_graph"
THREAD_ID_KEY = "agent_thread_id"
CHAT_ERROR_KEY = "chat_error"
STOP_FLAG_KEY = "chat_stop_flag"
RUN_ACTIVE_KEY = "chat_run_active"
RUN_HOLDER_KEY = "chat_run_holder"
_POLL_S = 0.3


def ensure_thread_id(
    session_state: Any, *, checkpoint_path: Path | None = None
) -> str:
    """Reuse the session thread id, or restore the one next to CHECKPOINT_PATH."""
    thread_id = session_state.get(THREAD_ID_KEY)
    if isinstance(thread_id, str) and thread_id.strip():
        return thread_id
    if checkpoint_path is not None:
        stored = load_persisted_thread_id(checkpoint_path)
        if stored:
            session_state[THREAD_ID_KEY] = stored
            return stored
    thread_id = str(uuid.uuid4())
    session_state[THREAD_ID_KEY] = thread_id
    if checkpoint_path is not None:
        persist_thread_id(checkpoint_path, thread_id)
    return thread_id


def start_new_chat(
    session_state: Any, *, checkpoint_path: Path | None = None
) -> str:
    """Mint a new thread id. The previous checkpoint is left unused."""
    request_chat_stop(session_state)
    session_state.pop(RUN_ACTIVE_KEY, None)
    session_state.pop(RUN_HOLDER_KEY, None)
    thread_id = str(uuid.uuid4())
    session_state[THREAD_ID_KEY] = thread_id
    session_state.pop(CHAT_ERROR_KEY, None)
    if checkpoint_path is not None:
        persist_thread_id(checkpoint_path, thread_id)
    return thread_id


def ensure_stop_flag(session_state: Any) -> threading.Event:
    """Session-lived stop flag. The compiled graph reads this Event."""
    flag = session_state.get(STOP_FLAG_KEY)
    if isinstance(flag, threading.Event):
        return flag
    flag = threading.Event()
    session_state[STOP_FLAG_KEY] = flag
    return flag


def request_chat_stop(session_state: Any) -> None:
    """Ask the graph not to start another tool. In-flight work may finish."""
    ensure_stop_flag(session_state).set()


def start_chat_run(
    session_state: Any, invoke_fn: Callable[[], dict[str, Any]]
) -> None:
    """Run invoke off the Streamlit thread so Stop can be clicked."""
    if session_state.get(RUN_ACTIVE_KEY):
        holder = session_state.get(RUN_HOLDER_KEY)
        if isinstance(holder, dict) and not holder.get("done"):
            return
    ensure_stop_flag(session_state).clear()
    holder: dict[str, Any] = {"result": None, "done": False}

    def _run() -> None:
        try:
            holder["result"] = invoke_fn()
        except Exception as exc:
            _LOG.info("UI chat invoke failed")
            holder["result"] = {
                "error": str(exc).strip() or "Chat failed.",
            }
        finally:
            holder["done"] = True

    session_state[RUN_HOLDER_KEY] = holder
    session_state[RUN_ACTIVE_KEY] = True
    threading.Thread(target=_run, daemon=True).start()


def finish_chat_run(session_state: Any) -> dict[str, Any] | None:
    """Return a finished background invoke result and clear the run keys."""
    if not session_state.get(RUN_ACTIVE_KEY):
        return None
    holder = session_state.get(RUN_HOLDER_KEY)
    if not isinstance(holder, dict) or not holder.get("done"):
        return None
    result = holder.get("result")
    session_state.pop(RUN_HOLDER_KEY, None)
    session_state[RUN_ACTIVE_KEY] = False
    if isinstance(result, dict):
        return result
    return {"error": "Chat failed."}


def ensure_graph(session_state: Any, settings: Settings) -> Any:
    """Reuse the compiled graph so the sqlite checkpointer stays open."""
    flag = ensure_stop_flag(session_state)
    graph = session_state.get(GRAPH_KEY)
    if graph is None:
        graph = build_graph(
            settings=settings,
            checkpointer=sqlite_checkpointer(settings.checkpoint_path),
            stop_requested=flag.is_set,
        )
        session_state[GRAPH_KEY] = graph
    return graph


def thread_config(thread_id: str) -> dict[str, Any]:
    """LangGraph checkpointer config for one conversation thread."""
    return {"configurable": {"thread_id": thread_id}}


def graph_snapshot(graph: Any, thread_id: str) -> Any:
    """One checkpointer read for this thread."""
    return graph.get_state(thread_config(thread_id))


def _snapshot_values(snap: Any) -> dict[str, Any]:
    values = getattr(snap, "values", None) or {}
    return values if isinstance(values, dict) else {}


def graph_messages(
    graph: Any, thread_id: str, *, snap: Any | None = None
) -> list[AgentMessage]:
    """Return checkpointed user/assistant messages. Not the system prompt."""
    state = snap if snap is not None else graph_snapshot(graph, thread_id)
    messages = _snapshot_values(state).get("messages") or []
    return list(messages)


def graph_error(
    graph: Any, thread_id: str, *, snap: Any | None = None
) -> str | None:
    """Return the last visible graph error, if any."""
    state = snap if snap is not None else graph_snapshot(graph, thread_id)
    error = _snapshot_values(state).get("error")
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


def graph_interrupt_payload(
    graph: Any, thread_id: str, *, snap: Any | None = None
) -> dict[str, Any] | None:
    """Return a known interrupt value, if the graph is paused."""
    state = snap if snap is not None else graph_snapshot(graph, thread_id)
    interrupts = getattr(state, "interrupts", ()) or ()
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    if isinstance(value, dict) and value.get("kind") in _INTERRUPT_KINDS:
        return value
    return None


def graph_profile_summary(
    graph: Any, thread_id: str, *, snap: Any | None = None
) -> dict[str, Any] | None:
    """Return the checkpointed compact profile, if one was written."""
    state = snap if snap is not None else graph_snapshot(graph, thread_id)
    summary = _snapshot_values(state).get("profile_summary")
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
    thread_id = ensure_thread_id(
        st.session_state, checkpoint_path=settings.checkpoint_path
    )
    finished = finish_chat_run(st.session_state)
    if finished is not None:
        store_invoke_result(st.session_state, finished)
        st.rerun()
    snap = graph_snapshot(graph, thread_id)
    for message in graph_messages(graph, thread_id, snap=snap):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        with st.chat_message(role):
            st.write(message.get("content", ""))
    values = _snapshot_values(snap)
    raw_artifacts = values.get("artifacts") or []
    artifact_paths = [
        item
        for item in raw_artifacts
        if isinstance(item, str) and item.strip()
    ]
    render_table_artifacts(
        artifact_paths, artifact_dir=settings.artifact_dir
    )
    render_chart_artifacts(
        artifact_paths, artifact_dir=settings.artifact_dir
    )
    render_report_download(
        values.get("last_tool_result"),
        artifact_paths,
        thread_id=thread_id,
    )
    error = visible_chat_error(
        graph_error(graph, thread_id, snap=snap), st.session_state
    )
    if error:
        st.error(error)
    run_active = bool(st.session_state.get(RUN_ACTIVE_KEY))
    pending = graph_interrupt_payload(graph, thread_id, snap=snap)
    if pending is not None and not run_active:
        kind = pending.get("kind")
        decision: dict[str, Any] | None = None
        caption = "Confirm, select, or abort the data source before chatting."
        if kind == KIND_CONFIRM_SOURCES:
            decision = render_source_confirm(pending)
        elif kind == KIND_PROFILE_STEP:
            decision = render_profile_pause(
                pending, graph_profile_summary(graph, thread_id, snap=snap)
            )
            caption = (
                "Continue, skip remaining, or abort profiling before chatting."
            )
        elif kind == KIND_APPROVE_CODE:
            decision = render_code_review(pending)
            caption = (
                "Edit, approve, or reject generated SQL or Python before chatting."
            )
        elif kind == KIND_APPROVE_LOAD:
            decision = render_load_pause(pending)
            caption = "Approve or reject an over-limit full-file load before chatting."
        if decision is not None:
            chosen = decision
            start_chat_run(
                st.session_state,
                lambda: resume_interrupt(
                    graph, decision=chosen, thread_id=thread_id
                ),
            )
            st.rerun()
        st.caption(caption)
        return
    if run_active:
        if st.button("Stop", key="chat_stop"):
            request_chat_stop(st.session_state)
        st.caption("Waiting for the local model… Stop skips the next tool.")
        time.sleep(_POLL_S)
        st.rerun()
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
    start_chat_run(
        st.session_state,
        lambda: invoke_user_turn(
            graph,
            user_text=typed,
            hitl_mode=str(mode),
            thread_id=thread_id,
            source_ids=source_ids,
        ),
    )
    st.rerun()
