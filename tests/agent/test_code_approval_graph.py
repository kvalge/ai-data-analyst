# test_code_approval_graph.py

"""Graph tests: Standard pauses before run_analysis_code; reject does not run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

from src.agent.audit import (
    DECISION_AUTO,
    OUTCOME_ERROR,
    OUTCOME_REJECTED,
    OUTCOME_SUCCESS,
    audit_log_path,
    default_audit_dir,
)
from src.agent.code_approval import (
    ACTION_APPROVE,
    ACTION_EDIT_RUN,
    ACTION_REJECT,
    KIND_APPROVE_CODE,
    TOOL_QUERY_DATABASE,
    TOOL_RUN_ANALYSIS_CODE,
)
from src.agent.graph import build_graph
from src.agent.state import (
    HITL_MODE_AUTO,
    HITL_MODE_STANDARD,
    AgentState,
    empty_agent_state,
)
from src.config import Settings, load_settings
from src.storage.registry import list_file_sources, save_file_source
from src.storage.sources import postgres_source_id
from src.ui.chat import graph_interrupt_payload
from tests.db.fake_postgres import FakeConnection, FakeCursor

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}

_THREAD = {"configurable": {"thread_id": "code-approval-thread"}}
_SQL_THREAD = {"configurable": {"thread_id": "code-approval-sql-thread"}}
_AUTO_THREAD = {"configurable": {"thread_id": "code-approval-auto-thread"}}
_INVALID_THREAD = {"configurable": {"thread_id": "code-approval-invalid-thread"}}
_PRINT = "print(1)\n"
_SELECT = "SELECT date, region, revenue FROM sales"


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


def _tool_json(source_id: str, code: str = _PRINT) -> str:
    return json.dumps(
        {
            "name": TOOL_RUN_ANALYSIS_CODE,
            "arguments": {"source_id": source_id, "code": code},
        }
    )


def _audit_record(settings: Settings) -> dict[str, Any]:
    path = audit_log_path(default_audit_dir(settings.upload_dir))
    return json.loads(path.read_text(encoding="utf-8").strip())


def _interrupt_value(result: dict[str, Any]) -> dict[str, Any]:
    items = result.get("__interrupt__") or []
    assert items, "expected an approve_code interrupt"
    value = items[0].value
    assert isinstance(value, dict)
    return value


def test_auto_runs_analysis_code_without_interrupt(settings: Settings):
    """Auto executes generated Python immediately and still writes the audit line."""
    source_id = _source_id(settings)
    replies = [_tool_json(source_id), "printed one"]

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return replies.pop(0)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(
        _user_turn("print one", hitl_mode=HITL_MODE_AUTO), _AUTO_THREAD
    )
    assert result.get("__interrupt__") is None
    assert result["error"] is None
    assert "1" in result["last_tool_result"]["stdout"]
    assert result["messages"][-1]["content"] == "printed one"
    record = _audit_record(settings)
    assert record["tool"] == TOOL_RUN_ANALYSIS_CODE
    assert record["source_id"] == source_id
    assert record["code"] == _PRINT.strip()
    assert record["decision"] == DECISION_AUTO
    assert record["outcome"] == OUTCOME_SUCCESS
    assert set(record) == {
        "timestamp",
        "tool",
        "source_id",
        "code",
        "decision",
        "outcome",
    }
    assert "rows" not in record
    assert "stdout" not in record


def test_standard_interrupts_before_run_analysis_code(settings: Settings):
    """Standard pauses with code, source_id, and rationale before the sandbox."""
    source_id = _source_id(settings)

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return _tool_json(source_id)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("sum revenue"), _THREAD)
    payload = _interrupt_value(result)
    assert payload["kind"] == KIND_APPROVE_CODE
    assert payload["tool"] == TOOL_RUN_ANALYSIS_CODE
    assert payload["code"] == _PRINT.strip()
    assert payload["source_id"] == source_id
    assert payload["rationale"] == ""
    assert result.get("last_tool_result") is None
    if settings.artifact_dir.is_dir():
        assert list(settings.artifact_dir.rglob("*.csv")) == []


def test_approve_runs_the_sandbox(settings: Settings):
    """Approve executes the original code and returns stdout."""
    source_id = _source_id(settings)
    replies = [_tool_json(source_id), "printed one"]

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return replies.pop(0)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("print one"), _THREAD)
    result = graph.invoke(Command(resume={"action": ACTION_APPROVE}), _THREAD)
    assert result.get("__interrupt__") is None
    assert result["error"] is None
    assert "1" in result["last_tool_result"]["stdout"]
    assert result["messages"][-1]["content"] == "printed one"


def test_edit_run_executes_edited_code(settings: Settings):
    """edit_run replaces the code; the original snippet is not what ran."""
    source_id = _source_id(settings)
    replies = [_tool_json(source_id, "print(0)\n"), "printed seven"]

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return replies.pop(0)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("print something"), _THREAD)
    result = graph.invoke(
        Command(resume={"action": ACTION_EDIT_RUN, "code": "print(7)\n"}),
        _THREAD,
    )
    assert result["error"] is None
    assert "7" in result["last_tool_result"]["stdout"]
    assert "0" not in result["last_tool_result"]["stdout"]


def test_reject_does_not_execute(settings: Settings):
    """Reject ends the turn without a sandbox result."""
    source_id = _source_id(settings)
    calls = 0

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return _tool_json(source_id)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("print one"), _THREAD)
    result = graph.invoke(Command(resume={"action": ACTION_REJECT}), _THREAD)
    assert calls == 1
    assert result["error"] == "Generated code was rejected."
    assert result["last_tool_result"] is None
    assert result["pending_tool"] is None
    if settings.artifact_dir.is_dir():
        assert list(settings.artifact_dir.rglob("*")) == []
    record = _audit_record(settings)
    assert record["decision"] == ACTION_REJECT
    assert record["outcome"] == OUTCOME_REJECTED
    assert record["code"] == _PRINT.strip()
    assert "stdout" not in record


def test_invalid_resume_audit_is_not_a_user_reject(settings: Settings):
    """A malformed resume does not run and is audited as error, not reject."""
    source_id = _source_id(settings)

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return _tool_json(source_id)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("print one"), _INVALID_THREAD)
    result = graph.invoke(Command(resume="nonsense"), _INVALID_THREAD)
    assert result["error"] == "Code approval decision is invalid."
    assert result["last_tool_result"] is None
    record = _audit_record(settings)
    assert record["decision"] == ACTION_REJECT
    assert record["outcome"] == OUTCOME_ERROR
    assert record["code"] == _PRINT.strip()
    assert "stdout" not in record


def test_edit_run_audit_records_edited_code(settings: Settings):
    """The audit line stores the edited text, not the original snippet."""
    source_id = _source_id(settings)
    replies = [_tool_json(source_id, "print(0)\n"), "printed seven"]

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return replies.pop(0)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("print something"), _THREAD)
    graph.invoke(
        Command(resume={"action": ACTION_EDIT_RUN, "code": "print(7)\n"}),
        _THREAD,
    )
    record = _audit_record(settings)
    assert record["decision"] == ACTION_EDIT_RUN
    assert record["outcome"] == OUTCOME_SUCCESS
    assert record["code"] == "print(7)"
    assert "print(0)" not in record["code"]


def test_sandbox_error_audit_records_error_outcome(settings: Settings):
    """A failed sandbox still writes code and outcome=error, not stdout."""
    source_id = _source_id(settings)

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return _tool_json(source_id, "print(1 / 0)\n")

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("divide"), _THREAD)
    result = graph.invoke(Command(resume={"action": ACTION_APPROVE}), _THREAD)
    assert result["error"]
    record = _audit_record(settings)
    assert record["decision"] == ACTION_APPROVE
    assert record["outcome"] == OUTCOME_ERROR
    assert record["code"] == "print(1 / 0)"
    assert "stdout" not in record
    assert "rows" not in record


def test_standard_interrupts_before_query_database(tmp_path: Path):
    """Standard pauses on SQL, then approve runs through a mocked cursor."""
    pg_settings = load_settings(
        environ={
            "OLLAMA_HOST": "http://ollama.test:11434",
            **_PLACEHOLDER_MODELS,
            "DB_HOST": "db.local",
            "DB_PORT": "5432",
            "DB_NAME": "analytics",
            "DB_USER": "reader",
            "DB_PASSWORD": "secret",
        },
        load_dotenv_file=False,
        project_root=tmp_path,
    )
    connection_id = postgres_source_id(pg_settings)
    cursor = FakeCursor([("2024-01-01", "North", 10.0)])
    replies = [
        json.dumps(
            {
                "name": TOOL_QUERY_DATABASE,
                "arguments": {"connection_id": connection_id, "sql": _SELECT},
            }
        ),
        "one row",
    ]

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return replies.pop(0)

    def fake_connect(**_kw: object) -> FakeConnection:
        return FakeConnection(cursor)

    graph = build_graph(
        settings=pg_settings,
        complete_fn=fake_complete,
        include_postgres=True,
        connect=fake_connect,
    )
    result = graph.invoke(_user_turn("query sales"), _SQL_THREAD)
    payload = _interrupt_value(result)
    assert payload["kind"] == KIND_APPROVE_CODE
    assert payload["tool"] == TOOL_QUERY_DATABASE
    assert payload["code"] == _SELECT
    assert payload["source_id"] == connection_id
    assert result.get("last_tool_result") is None
    assert cursor.sql is None

    result = graph.invoke(
        Command(resume={"action": ACTION_APPROVE}), _SQL_THREAD
    )
    assert result.get("__interrupt__") is None
    assert result["error"] is None
    assert cursor.sql == _SELECT
    assert result["last_tool_result"]["connection_id"] == connection_id
    assert result["last_tool_result"]["columns"] == ["date", "region", "revenue"]
    assert result["messages"][-1]["content"] == "one row"


def test_chat_helper_reads_approve_code_interrupt(settings: Settings):
    """graph_interrupt_payload accepts an approve_code pause."""
    source_id = _source_id(settings)

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return _tool_json(source_id)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("print one"), _THREAD)
    payload = graph_interrupt_payload(graph, "code-approval-thread")
    assert payload is not None
    assert payload["kind"] == KIND_APPROVE_CODE
    assert payload["code"] == _PRINT.strip()
