# test_graph.py

"""Tests for the LangGraph agent with mocked LLM and allowlisted tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agent.graph import build_graph
from src.agent.json_output import STRICT_RETRY_INSTRUCTION
from src.agent.llm import LlmError
from src.agent.prompts import build_system_prompt
from src.agent.state import HITL_MODE_STANDARD, AgentState, empty_agent_state
from src.config import Settings, load_settings
from src.storage.registry import save_file_source

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}

_THREAD = {"configurable": {"thread_id": "test-thread"}}
_GRAPH_PATH = Path(__file__).resolve().parents[1] / "src" / "agent" / "graph.py"


@pytest.fixture
def settings(tmp_path, sample_sales_csv: Path) -> Settings:
    """Frozen settings with one registered sales file so confirm_sources can pass."""
    loaded = load_settings(
        environ={"OLLAMA_HOST": "http://ollama.test:11434", **_PLACEHOLDER_MODELS},
        load_dotenv_file=False,
        project_root=tmp_path,
    )
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    save_file_source(incoming, loaded.upload_dir, original_name="sales.csv")
    return loaded


def _user_turn(text: str) -> AgentState:
    state = empty_agent_state()
    state["messages"] = [{"role": "user", "content": text}]
    return state


def test_agent_appends_plain_text_reply(settings: Settings):
    """A plain-text model reply skips execute_tool."""

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return "plain reply"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("hello"), _THREAD)
    assert result["messages"][-1] == {"role": "assistant", "content": "plain reply"}
    assert result["last_tool_result"] is None


def test_agent_sends_system_prompt_to_llm(settings: Settings):
    """The LLM prompt starts with the mode-aware system prompt, not UI text."""
    captured: list[str] = []

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        captured.append(prompt)
        return "ok"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("hello"), _THREAD)
    expected = build_system_prompt(hitl_mode=HITL_MODE_STANDARD)
    assert captured[0].startswith(expected)
    assert "user: hello" in captured[0]


def test_agent_passes_settings_into_complete(settings: Settings):
    """The model name comes from settings, not a literal in the graph."""
    seen: list[Settings] = []

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        seen.append(kwargs["settings"])
        return "ok"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("hello"), _THREAD)
    assert seen[0].ollama_model_primary == settings.ollama_model_primary


def test_llm_error_is_visible_without_a_guessed_reply(settings: Settings):
    """A failed LLM call sets error and does not invent assistant text."""

    def boom(prompt: str, **kwargs: Any) -> str:
        raise LlmError("Could not reach Ollama.")

    graph = build_graph(settings=settings, complete_fn=boom)
    result = graph.invoke(_user_turn("hello"), _THREAD)
    assert result["error"] == "Could not reach Ollama."
    assert [m["role"] for m in result["messages"]] == ["user"]


def test_empty_llm_reply_is_visible_without_a_guessed_message(settings: Settings):
    """A blank model reply is a missing response, not an empty assistant turn."""

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return ""

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("hello"), _THREAD)
    assert result["error"] == "Ollama response was missing."
    assert [m["role"] for m in result["messages"]] == ["user"]


def test_missing_user_message_sets_error(settings: Settings):
    """The stub does not call the LLM when there is nothing to answer."""
    called = False

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        nonlocal called
        called = True
        return "should not run"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(empty_agent_state(), _THREAD)
    assert called is False
    assert result["error"] == "No user message to reply to."


def test_memory_saver_keeps_thread_history(settings: Settings):
    """A second turn on the same thread still sees the first messages."""
    replies = ["first-reply", "second-reply"]

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return replies.pop(0)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("one"), _THREAD)
    graph.invoke({"messages": [{"role": "user", "content": "two"}]}, _THREAD)
    snap = graph.get_state(_THREAD)
    contents = [m["content"] for m in snap.values["messages"]]
    assert contents == ["one", "first-reply", "two", "second-reply"]


def test_graph_module_does_not_import_streamlit():
    """Chat wiring stays out of this module."""
    source = _GRAPH_PATH.read_text(encoding="utf-8")
    assert "streamlit" not in source.lower()


def test_list_available_sources_runs_then_replies(
    settings: Settings, tmp_path, sample_sales_csv: Path
):
    """A mocked tool JSON runs list_available_sources, then a text reply."""
    upload_dir = settings.upload_dir
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(incoming, upload_dir, original_name="sales.csv")
    replies = [
        '{"name": "list_available_sources", "arguments": {}}',
        "There is 1 source.",
    ]
    prompts: list[str] = []

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        prompts.append(prompt)
        return replies.pop(0)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("what sources do I have?"), _THREAD)
    assert "Last tool result" in prompts[1]
    assert saved.source_id in prompts[1]
    assert result["error"] is None
    assert result["pending_tool"] is None
    sources = result["last_tool_result"]["sources"]
    assert len(sources) == 1
    assert sources[0]["source_id"] == saved.source_id
    assert "date,region,revenue" not in str(result["last_tool_result"])
    assert result["messages"][-1] == {
        "role": "assistant",
        "content": "There is 1 source.",
    }


def test_failed_tool_is_visible_without_a_second_llm_call(settings: Settings):
    """A domain tool error ends the turn. The model does not invent a result."""
    calls = 0

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return (
            '{"name": "read_file_sample",'
            ' "arguments": {"source_id": "file-missing"}}'
        )

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("show a sample"), _THREAD)
    assert calls == 1
    assert result["error"] == "Unknown source_id: file-missing"
    assert result["last_tool_result"] is None
    assert [m["role"] for m in result["messages"]] == ["user"]


def test_unknown_tool_retries_then_fails(settings: Settings):
    """An unknown tool name retries once, then fails without executing."""
    prompts: list[str] = []

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        prompts.append(prompt)
        return '{"name": "not_a_tool", "arguments": {}}'

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("list sources"), _THREAD)
    assert len(prompts) == 2
    assert STRICT_RETRY_INSTRUCTION in prompts[1]
    assert "Unknown tool: not_a_tool." in prompts[1]
    assert result["error"] == "Unknown tool: not_a_tool."
    assert result["last_tool_result"] is None
    assert result["pending_tool"] is None
    assert [m["role"] for m in result["messages"]] == ["user"]


def test_unknown_tool_retries_then_runs(
    settings: Settings, tmp_path, sample_sales_csv: Path
):
    """A valid tool JSON on the strict retry is executed, not guessed."""
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="sales.csv"
    )
    replies = [
        '{"name": "not_a_tool", "arguments": {}}',
        '{"name": "list_available_sources", "arguments": {}}',
        "There is 1 source.",
    ]

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return replies.pop(0)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("what sources do I have?"), _THREAD)
    assert result["error"] is None
    assert result["last_tool_result"]["sources"][0]["source_id"] == saved.source_id
    assert result["messages"][-1]["content"] == "There is 1 source."


def test_bad_args_retry_then_fail(settings: Settings):
    """Missing required arguments retry once, then fail without a guessed id."""
    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return '{"name": "read_file_sample", "arguments": {}}'

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("show a sample"), _THREAD)
    assert result["error"] == "Missing keys: source_id"
    assert result["last_tool_result"] is None
    assert [m["role"] for m in result["messages"]] == ["user"]
