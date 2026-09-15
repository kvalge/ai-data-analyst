# test_load_approval_graph.py

"""Graph tests: over-limit load_full_file pauses; reject does not load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

from src.agent.audit import (
    OUTCOME_REJECTED,
    audit_log_path,
    default_audit_dir,
)
from src.agent.graph import build_graph
from src.agent.load_approval import (
    ACTION_APPROVE,
    ACTION_REJECT,
    KIND_APPROVE_LOAD,
)
from src.agent.state import HITL_MODE_AUTO, HITL_MODE_STANDARD, AgentState
from src.agent.state import empty_agent_state
from src.config import Settings, load_settings
from src.storage.registry import list_file_sources, save_file_source
from src.tools.load_full_file import STATUS_LOADED
from src.ui.chat import graph_interrupt_payload

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}

_THREAD = {"configurable": {"thread_id": "load-approval-thread"}}
_AUTO_THREAD = {"configurable": {"thread_id": "load-approval-auto-thread"}}
_UNDER_THREAD = {"configurable": {"thread_id": "load-approval-under-thread"}}


@pytest.fixture
def settings(tmp_path, sample_sales_csv: Path) -> Settings:
    """Frozen settings with a 5-row sales file and a 3-row full-load cap."""
    loaded = load_settings(
        environ={
            "OLLAMA_HOST": "http://ollama.test:11434",
            "MAX_FULL_LOAD_ROWS": "3",
            **_PLACEHOLDER_MODELS,
        },
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


def _tool_json(source_id: str) -> str:
    return json.dumps(
        {"name": "load_full_file", "arguments": {"source_id": source_id}}
    )


def _interrupt_value(result: dict[str, Any]) -> dict[str, Any]:
    items = result.get("__interrupt__") or []
    assert items, "expected an approve_load interrupt"
    value = items[0].value
    assert isinstance(value, dict)
    return value


def test_over_limit_interrupts_before_load(settings: Settings):
    """A 5-row file over a 3-row cap pauses with limits, not rows."""
    source_id = _source_id(settings)

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return _tool_json(source_id)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("load the file"), _THREAD)
    payload = _interrupt_value(result)
    assert payload["kind"] == KIND_APPROVE_LOAD
    assert payload["source_id"] == source_id
    assert payload["reason"] == "row_count"
    assert payload["row_count"] == 5
    assert payload["max_full_load_rows"] == 3
    assert "columns" not in payload
    assert result.get("last_tool_result") is None


def test_approve_loads_over_limit(settings: Settings):
    """Approve injects allow_over_limit and returns loaded metadata."""
    source_id = _source_id(settings)
    replies = [_tool_json(source_id), "loaded"]

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return replies.pop(0)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("load the file"), _THREAD)
    result = graph.invoke(Command(resume={"action": ACTION_APPROVE}), _THREAD)
    assert result.get("__interrupt__") is None
    assert result["error"] is None
    loaded = result["last_tool_result"]
    assert loaded["status"] == STATUS_LOADED
    assert loaded["columns"] == ["date", "region", "revenue"]
    assert loaded["row_count"] == 5
    assert "rows" not in loaded
    assert result["messages"][-1]["content"] == "loaded"


def test_reject_does_not_load(settings: Settings):
    """Reject ends the turn without a loaded result."""
    source_id = _source_id(settings)
    calls = 0

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return _tool_json(source_id)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("load the file"), _THREAD)
    result = graph.invoke(Command(resume={"action": ACTION_REJECT}), _THREAD)
    assert calls == 1
    assert result["error"] == "Full-file load was rejected."
    assert result["last_tool_result"] is None
    assert result["pending_tool"] is None


def test_reject_writes_audit_line(settings: Settings):
    """A rejected over-limit load is audited; the pause itself is not a success."""
    source_id = _source_id(settings)

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return _tool_json(source_id)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("load the file"), _THREAD)
    graph.invoke(Command(resume={"action": ACTION_REJECT}), _THREAD)
    path = audit_log_path(default_audit_dir(settings.upload_dir))
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    loads = [row for row in records if row["tool"] == "load_full_file"]
    assert len(loads) == 1
    assert loads[0]["source_id"] == source_id
    assert loads[0]["decision"] == ACTION_REJECT
    assert loads[0]["outcome"] == OUTCOME_REJECTED
    assert loads[0]["code"] is None


def test_auto_still_pauses_over_limit(settings: Settings):
    """Auto skips code approval but still pauses an over-limit full load."""
    source_id = _source_id(settings)

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return _tool_json(source_id)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(
        _user_turn("load the file", hitl_mode=HITL_MODE_AUTO), _AUTO_THREAD
    )
    payload = _interrupt_value(result)
    assert payload["kind"] == KIND_APPROVE_LOAD
    assert result.get("last_tool_result") is None


def test_under_limit_does_not_interrupt(tmp_path: Path, sample_sales_csv: Path):
    """A file under both caps loads without an approve_load pause."""
    loaded = load_settings(
        environ={
            "OLLAMA_HOST": "http://ollama.test:11434",
            "MAX_FULL_LOAD_ROWS": "10",
            **_PLACEHOLDER_MODELS,
        },
        load_dotenv_file=False,
        project_root=tmp_path,
    )
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    save_file_source(incoming, loaded.upload_dir, original_name="sales.csv")
    source_id = list_file_sources(loaded.upload_dir)[0].source_id
    replies = [_tool_json(source_id), "under limit"]

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return replies.pop(0)

    graph = build_graph(settings=loaded, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("load the file"), _UNDER_THREAD)
    assert result.get("__interrupt__") is None
    assert result["last_tool_result"]["status"] == STATUS_LOADED
    assert result["messages"][-1]["content"] == "under limit"


def test_chat_helper_reads_approve_load_interrupt(settings: Settings):
    """graph_interrupt_payload accepts an approve_load pause."""
    source_id = _source_id(settings)

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return _tool_json(source_id)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("load the file"), _THREAD)
    payload = graph_interrupt_payload(graph, "load-approval-thread")
    assert payload is not None
    assert payload["kind"] == KIND_APPROVE_LOAD
