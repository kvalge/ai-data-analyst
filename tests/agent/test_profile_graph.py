# test_profile_graph.py

"""Graph tests: Standard runs three profile nodes; Guided interrupts three times."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.profile_steps import (
    ACTION_ABORT,
    ACTION_CONTINUE,
    ACTION_SKIP_REMAINING,
    KIND_PROFILE_STEP,
    SECTION_DQ,
    SECTION_EDA,
    SECTION_SCHEMA,
)
from src.agent.state import HITL_MODE_GUIDED, HITL_MODE_STANDARD, AgentState
from src.agent.state import empty_agent_state
from src.config import Settings, load_settings
from src.storage.registry import list_file_sources, save_file_source
from src.storage.sources import make_postgres_source
from src.ui.chat import graph_interrupt_payload

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}

_THREAD = {"configurable": {"thread_id": "profile-thread"}}


@pytest.fixture
def settings(tmp_path, sample_sales_csv: Path) -> Settings:
    """Frozen settings with one registered sales file."""
    loaded = load_settings(
        environ={"OLLAMA_HOST": "http://ollama.test:11434", **_PLACEHOLDER_MODELS},
        load_dotenv_file=False,
        project_root=tmp_path,
    )
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    save_file_source(incoming, loaded.upload_dir, original_name="sales.csv")
    return loaded


def _source_id(settings: Settings) -> str:
    return list_file_sources(settings.upload_dir)[0].source_id


def _user_turn(text: str, *, hitl_mode: str = HITL_MODE_STANDARD) -> AgentState:
    state = empty_agent_state(hitl_mode=hitl_mode)
    state["messages"] = [{"role": "user", "content": text}]
    return state


def _interrupt_value(result: dict[str, Any]) -> dict[str, Any]:
    items = result.get("__interrupt__") or []
    assert items, "expected a profile_step interrupt"
    value = items[0].value
    assert isinstance(value, dict)
    return value


def test_standard_runs_three_profile_nodes_without_interrupt(settings: Settings):
    """Standard streams detect_schema, run_dq, and run_eda with no pause."""

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return "plain reply"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    names: list[str] = []
    for chunk in graph.stream(
        _user_turn("hello"), _THREAD, stream_mode="updates"
    ):
        names.extend(chunk.keys())
    assert "detect_schema" in names
    assert "run_dq" in names
    assert "run_eda" in names
    assert "__interrupt__" not in names
    snap = graph.get_state(_THREAD)
    assert snap.values["error"] is None
    assert snap.values["profile_summary"]["source_id"] == _source_id(settings)
    assert _source_id(settings) in snap.values["cleared_profile_source_ids"]


def test_guided_interrupts_three_times(settings: Settings):
    """Guided pauses after schema, DQ, and EDA, then the agent replies."""
    called = 0

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        nonlocal called
        called += 1
        return "plain reply"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("hello", hitl_mode=HITL_MODE_GUIDED), _THREAD)
    first = _interrupt_value(result)
    assert first["kind"] == KIND_PROFILE_STEP
    assert first["step"] == SECTION_SCHEMA
    assert first["source_id"] == _source_id(settings)
    assert "rows" not in first
    assert called == 0

    result = graph.invoke(
        Command(resume={"action": ACTION_CONTINUE}), _THREAD
    )
    assert _interrupt_value(result)["step"] == SECTION_DQ
    assert called == 0

    result = graph.invoke(
        Command(resume={"action": ACTION_CONTINUE}), _THREAD
    )
    assert _interrupt_value(result)["step"] == SECTION_EDA
    assert called == 0

    result = graph.invoke(
        Command(resume={"action": ACTION_CONTINUE}), _THREAD
    )
    assert result.get("__interrupt__") is None
    assert result["error"] is None
    assert result["messages"][-1] == {
        "role": "assistant",
        "content": "plain reply",
    }
    assert called == 1
    assert _source_id(settings) in result["cleared_profile_source_ids"]


def test_guided_skip_remaining_after_schema_skips_later_pauses(settings: Settings):
    """Skip remaining after schema goes to the agent without DQ/EDA pauses."""
    called = 0

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        nonlocal called
        called += 1
        return "plain reply"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("hello", hitl_mode=HITL_MODE_GUIDED), _THREAD)
    assert _interrupt_value(result)["step"] == SECTION_SCHEMA
    result = graph.invoke(
        Command(resume={"action": ACTION_SKIP_REMAINING}), _THREAD
    )
    assert result.get("__interrupt__") is None
    assert result["error"] is None
    assert called == 1
    assert result["messages"][-1]["content"] == "plain reply"
    assert _source_id(settings) in result["cleared_profile_source_ids"]


def test_guided_abort_does_not_call_llm(settings: Settings):
    """Abort after schema ends the turn without a guessed reply."""
    called = 0

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        nonlocal called
        called += 1
        return "should not run"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("hello", hitl_mode=HITL_MODE_GUIDED), _THREAD)
    result = graph.invoke(Command(resume={"action": ACTION_ABORT}), _THREAD)
    assert result["error"] == "Profiling was aborted."
    assert called == 0
    assert [m["role"] for m in result["messages"]] == ["user"]
    assert _source_id(settings) not in (result.get("cleared_profile_source_ids") or [])


def test_second_guided_turn_does_not_reinterrupt(settings: Settings):
    """A later turn on the same source does not pause profiling again."""

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return "plain reply"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("hello", hitl_mode=HITL_MODE_GUIDED), _THREAD)
    graph.invoke(Command(resume={"action": ACTION_SKIP_REMAINING}), _THREAD)
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "again"}],
            "hitl_mode": HITL_MODE_GUIDED,
            "source_ids": [_source_id(settings)],
        },
        _THREAD,
    )
    assert result.get("__interrupt__") is None
    assert result["messages"][-1]["content"] == "plain reply"


def test_postgres_source_skips_profiling_and_reaches_agent(tmp_path):
    """Postgres is not profiled; the chat continues to the agent."""
    loaded = load_settings(
        environ={
            "OLLAMA_HOST": "http://ollama.test:11434",
            **_PLACEHOLDER_MODELS,
            "DB_NAME": "analytics",
            "DB_USER": "reader",
        },
        load_dotenv_file=False,
        project_root=tmp_path,
    )
    source = make_postgres_source(loaded)
    called = 0

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        nonlocal called
        called += 1
        return "plain reply"

    graph = build_graph(
        settings=loaded, complete_fn=fake_complete, include_postgres=True
    )
    state = _user_turn("hello")
    state["source_ids"] = [source.source_id]
    result = graph.invoke(state, _THREAD)
    assert result.get("__interrupt__") is None
    assert result["error"] is None
    assert result["profile_summary"] is None
    assert called == 1
    assert result["messages"][-1]["content"] == "plain reply"


def test_chat_helper_reads_profile_step_interrupt(settings: Settings):
    """graph_interrupt_payload accepts a profile_step pause, not only confirm."""

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return "plain reply"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("hello", hitl_mode=HITL_MODE_GUIDED), _THREAD)
    payload = graph_interrupt_payload(graph, "profile-thread")
    assert payload is not None
    assert payload["kind"] == KIND_PROFILE_STEP
    assert payload["step"] == SECTION_SCHEMA
