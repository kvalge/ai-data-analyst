# test_chat.py

"""Tests for chat helpers: graph turns, no system prompt in the UI layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agent.graph import build_graph
from src.agent.llm import LlmError
from src.agent.state import HITL_MODE_STANDARD
from src.config import Settings, load_settings
from src.storage.registry import save_file_source
from src.agent.confirm_sources import REASON_NO_SOURCES
from src.ui.chat import (
    ensure_thread_id,
    graph_error,
    graph_interrupt_payload,
    graph_messages,
    invoke_user_turn,
    store_invoke_result,
    visible_chat_error,
)

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}

_CHAT_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "chat.py"
_APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"


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


@pytest.fixture
def graph(settings: Settings):
    """A graph with a mocked LLM that replies in plain text."""

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return "plain reply"

    return build_graph(settings=settings, complete_fn=fake_complete)


def test_first_turn_appends_user_and_assistant(graph):
    """The UI helper sends user text and stores the model reply."""
    result = invoke_user_turn(
        graph,
        user_text="hello",
        hitl_mode=HITL_MODE_STANDARD,
        thread_id="chat-1",
    )
    assert [m["role"] for m in result["messages"]] == ["user", "assistant"]
    assert result["messages"][0]["content"] == "hello"
    assert result["messages"][1]["content"] == "plain reply"


def test_transcript_omits_system_prompt(graph):
    """Checkpointed messages are user/assistant only."""
    invoke_user_turn(
        graph,
        user_text="hello",
        hitl_mode=HITL_MODE_STANDARD,
        thread_id="chat-2",
    )
    roles = [m["role"] for m in graph_messages(graph, "chat-2")]
    assert "system" not in roles
    assert roles == ["user", "assistant"]


def test_chat_turn_without_sources_pauses(tmp_path):
    """invoke_user_turn surfaces the confirm_sources interrupt, not a guess."""
    empty = load_settings(
        environ={"OLLAMA_HOST": "http://ollama.test:11434", **_PLACEHOLDER_MODELS},
        load_dotenv_file=False,
        project_root=tmp_path,
    )

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return "should not run"

    graph = build_graph(settings=empty, complete_fn=fake_complete)
    invoke_user_turn(
        graph,
        user_text="hello",
        hitl_mode=HITL_MODE_STANDARD,
        thread_id="chat-none",
    )
    payload = graph_interrupt_payload(graph, "chat-none")
    assert payload is not None
    assert payload["reason"] == REASON_NO_SOURCES


def test_blank_chat_text_is_rejected(graph):
    """Whitespace is not sent to the graph."""
    with pytest.raises(ValueError, match="Chat text is empty"):
        invoke_user_turn(
            graph,
            user_text="   ",
            hitl_mode=HITL_MODE_STANDARD,
            thread_id="chat-3",
        )


def test_second_turn_keeps_history(graph):
    """The same thread id reuses MemorySaver history."""
    invoke_user_turn(
        graph,
        user_text="one",
        hitl_mode=HITL_MODE_STANDARD,
        thread_id="chat-4",
    )
    invoke_user_turn(
        graph,
        user_text="two",
        hitl_mode=HITL_MODE_STANDARD,
        thread_id="chat-4",
    )
    contents = [m["content"] for m in graph_messages(graph, "chat-4")]
    assert contents == ["one", "plain reply", "two", "plain reply"]


def test_graph_error_is_readable(settings: Settings):
    """A failed turn leaves a visible error and no assistant guess."""

    def fail(prompt: str, **kwargs: Any) -> str:
        raise LlmError("Could not reach Ollama.")

    graph = build_graph(settings=settings, complete_fn=fail)
    invoke_user_turn(
        graph,
        user_text="hello",
        hitl_mode=HITL_MODE_STANDARD,
        thread_id="chat-5",
    )
    assert graph_error(graph, "chat-5") == "Could not reach Ollama."
    assert [m["role"] for m in graph_messages(graph, "chat-5")] == ["user"]


def test_invoke_exception_becomes_visible_error():
    """A crashing invoke is a chat error, not an uncaught traceback."""

    class BoomGraph:
        def get_state(self, config: object) -> object:
            class Snap:
                values = {"messages": [], "error": None}

            return Snap()

        def invoke(self, payload: object, config: object) -> dict[str, Any]:
            raise ConnectionError("connection refused")

    result = invoke_user_turn(
        BoomGraph(),
        user_text="hello",
        hitl_mode=HITL_MODE_STANDARD,
        thread_id="chat-6",
    )
    assert result["error"] == "connection refused"
    assert result["messages"] == []


def test_escaped_invoke_error_is_shown_when_checkpoint_is_empty():
    """render_chat must use the invoke dict; the checkpoint has nothing to re-read."""

    class BoomGraph:
        def get_state(self, config: object) -> object:
            class Snap:
                values = {"messages": [], "error": None}

            return Snap()

        def invoke(self, payload: object, config: object) -> dict[str, Any]:
            raise ConnectionError("connection refused")

    graph = BoomGraph()
    result = invoke_user_turn(
        graph,
        user_text="hello",
        hitl_mode=HITL_MODE_STANDARD,
        thread_id="chat-7",
    )
    session: dict[str, object] = {}
    assert graph_error(graph, "chat-7") is None
    store_invoke_result(session, result)
    assert visible_chat_error(graph_error(graph, "chat-7"), session) == (
        "connection refused"
    )


def test_ensure_thread_id_reuses_existing():
    """A session keeps one thread id."""
    session: dict[str, object] = {"agent_thread_id": "kept"}
    assert ensure_thread_id(session) == "kept"


def test_chat_module_does_not_import_system_prompt():
    """System prompt text stays out of the chat UI module."""
    source = _CHAT_PATH.read_text(encoding="utf-8")
    assert "build_system_prompt" not in source
    assert "build_prompt" not in source
    assert "You are a local data-analyst" not in source


def test_app_module_does_not_import_system_prompt():
    """The Streamlit entry point also must not own the system prompt."""
    source = _APP_PATH.read_text(encoding="utf-8")
    assert "build_system_prompt" not in source
    assert "You are a local data-analyst" not in source
