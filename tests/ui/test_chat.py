# test_chat.py

"""Tests for chat helpers: graph turns, no system prompt in the UI layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agent.checkpoint import load_persisted_thread_id, persist_thread_id
from src.agent.confirm_sources import REASON_NO_SOURCES
from src.agent.graph import build_graph
from src.agent.llm import LlmError
from src.agent.state import HITL_MODE_STANDARD
from src.config import Settings, load_settings
from src.storage.registry import save_file_source
from src.ui.chat import (
    CHAT_ERROR_KEY,
    THREAD_ID_KEY,
    ensure_thread_id,
    graph_error,
    graph_interrupt_payload,
    graph_messages,
    graph_profile_summary,
    graph_snapshot,
    invoke_user_turn,
    start_new_chat,
    store_invoke_result,
    visible_chat_error,
)

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}

_CHAT_PATH = Path(__file__).resolve().parents[2] / "src" / "ui" / "chat.py"
_APP_PATH = Path(__file__).resolve().parents[2] / "src" / "ui" / "app.py"


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


class _BoomGraph:
    """Graph stub: empty checkpoint; invoke always raises."""

    def get_state(self, config: object) -> object:
        class Snap:
            values = {"messages": [], "error": None}

        return Snap()

    def invoke(self, payload: object, config: object) -> dict[str, Any]:
        raise ConnectionError("connection refused")


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


def test_malformed_json_after_retry_is_visible_in_chat(settings: Settings):
    """After one parse retry the error is in chat, not an invented assistant turn."""

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return "{not-json"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    invoke_user_turn(
        graph,
        user_text="hello",
        hitl_mode=HITL_MODE_STANDARD,
        thread_id="chat-retry",
    )
    assert graph_error(graph, "chat-retry") == "Could not parse JSON."
    assert [m["role"] for m in graph_messages(graph, "chat-retry")] == ["user"]


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
    result = invoke_user_turn(
        _BoomGraph(),
        user_text="hello",
        hitl_mode=HITL_MODE_STANDARD,
        thread_id="chat-6",
    )
    assert result["error"] == "connection refused"
    assert result["messages"] == []


def test_escaped_invoke_error_is_shown_when_checkpoint_is_empty():
    """render_chat must use the invoke dict; the checkpoint has nothing to re-read."""
    graph = _BoomGraph()
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
    session: dict[str, object] = {THREAD_ID_KEY: "kept"}
    assert ensure_thread_id(session) == "kept"


def test_ensure_thread_id_mints_when_missing():
    """A new session gets a thread id and then reuses it."""
    session: dict[str, object] = {}
    first = ensure_thread_id(session)
    assert first
    assert session[THREAD_ID_KEY] == first
    assert ensure_thread_id(session) == first


def test_start_new_chat_replaces_thread_and_clears_error():
    """New chat is a new thread id. A leftover chat error does not follow."""
    session: dict[str, object] = {
        THREAD_ID_KEY: "old-thread",
        CHAT_ERROR_KEY: "connection refused",
    }
    new_id = start_new_chat(session)
    assert new_id != "old-thread"
    assert session[THREAD_ID_KEY] == new_id
    assert CHAT_ERROR_KEY not in session


def test_ensure_thread_id_restores_sidecar(tmp_path: Path):
    """A fresh session picks up the thread id written next to CHECKPOINT_PATH."""
    path = tmp_path / "graph.sqlite"
    persist_thread_id(path, "stored-thread")
    session: dict[str, object] = {}
    assert ensure_thread_id(session, checkpoint_path=path) == "stored-thread"
    assert session[THREAD_ID_KEY] == "stored-thread"


def test_ensure_thread_id_writes_sidecar_when_minting(tmp_path: Path):
    """The first session on a path records the minted thread id."""
    path = tmp_path / "graph.sqlite"
    session: dict[str, object] = {}
    minted = ensure_thread_id(session, checkpoint_path=path)
    assert load_persisted_thread_id(path) == minted


def test_start_new_chat_writes_sidecar(tmp_path: Path):
    """New chat replaces the stored thread so a restart opens the new one."""
    path = tmp_path / "graph.sqlite"
    persist_thread_id(path, "old-thread")
    session: dict[str, object] = {THREAD_ID_KEY: "old-thread"}
    new_id = start_new_chat(session, checkpoint_path=path)
    assert load_persisted_thread_id(path) == new_id
    assert new_id != "old-thread"


def test_helpers_reuse_one_snapshot():
    """Readers do not hit the checkpointer again when a snapshot is passed."""

    class _CountGraph:
        def __init__(self) -> None:
            self.calls = 0

        def get_state(self, config: object) -> object:
            self.calls += 1

            class Snap:
                values = {"messages": [], "error": None, "profile_summary": None}
                interrupts = ()

            return Snap()

    graph = _CountGraph()
    snap = graph_snapshot(graph, "chat-once")
    graph_messages(graph, "chat-once", snap=snap)
    graph_error(graph, "chat-once", snap=snap)
    graph_interrupt_payload(graph, "chat-once", snap=snap)
    graph_profile_summary(graph, "chat-once", snap=snap)
    assert graph.calls == 1


def test_new_thread_id_does_not_see_prior_messages(graph):
    """thread_id in the graph config isolates one conversation from another."""
    invoke_user_turn(
        graph,
        user_text="one",
        hitl_mode=HITL_MODE_STANDARD,
        thread_id="chat-kept",
    )
    assert graph_messages(graph, "chat-fresh") == []
    assert [m["content"] for m in graph_messages(graph, "chat-kept")] == [
        "one",
        "plain reply",
    ]


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
