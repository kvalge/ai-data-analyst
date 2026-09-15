# test_checkpoint.py

"""Sqlite checkpointer and the thread-id sidecar next to CHECKPOINT_PATH."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.checkpoint import (
    load_persisted_thread_id,
    persist_thread_id,
    sqlite_checkpointer,
    thread_id_path,
)
from src.agent.graph import build_graph
from src.agent.state import empty_agent_state
from src.config import Settings, load_settings
from src.storage.registry import save_file_source

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}

_THREAD_ID = "sqlite-thread"
_THREAD = {"configurable": {"thread_id": _THREAD_ID}}


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


def _user_turn(text: str):
    state = empty_agent_state()
    state["messages"] = [{"role": "user", "content": text}]
    return state


def test_thread_id_path_is_sidecar_not_sqlite(tmp_path: Path):
    """The current thread lives beside the sqlite file, not inside it."""
    sqlite_path = tmp_path / "checkpoints" / "graph.sqlite"
    assert thread_id_path(sqlite_path) == tmp_path / "checkpoints" / "graph.thread"


def test_load_persisted_thread_id_missing(tmp_path: Path):
    """No sidecar means no stored thread."""
    assert load_persisted_thread_id(tmp_path / "graph.sqlite") is None


def test_load_persisted_thread_id_blank(tmp_path: Path):
    """Whitespace-only sidecar is treated as missing."""
    path = tmp_path / "graph.sqlite"
    thread_id_path(path).write_text("  \n", encoding="utf-8")
    assert load_persisted_thread_id(path) is None


def test_persist_then_load_thread_id(tmp_path: Path):
    """A written thread id is readable after a new process would start."""
    path = tmp_path / "nested" / "graph.sqlite"
    persist_thread_id(path, "  thread-abc  ")
    assert load_persisted_thread_id(path) == "thread-abc"


def test_build_graph_defaults_to_memory_saver(settings: Settings):
    """Unit tests keep MemorySaver unless they pass a checkpointer."""

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return "ok"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    assert isinstance(graph.checkpointer, MemorySaver)


def test_sqlite_checkpointer_uses_wal(tmp_path: Path):
    """Two Streamlit sessions share one file; WAL is set so readers are not locked out."""
    saver = sqlite_checkpointer(tmp_path / "graph.sqlite")
    try:
        mode = saver.conn.execute("PRAGMA journal_mode").fetchone()
        timeout = saver.conn.execute("PRAGMA busy_timeout").fetchone()
        assert mode is not None and mode[0].lower() == "wal"
        assert timeout == (5000,)
    finally:
        saver.conn.close()


def test_sqlite_round_trip_restores_messages(settings: Settings, tmp_path: Path):
    """A new graph on a new connection sees the same thread's messages."""

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return "plain reply"

    path = tmp_path / "graph.sqlite"
    first = sqlite_checkpointer(path)
    assert isinstance(first, SqliteSaver)
    try:
        graph = build_graph(
            settings=settings,
            complete_fn=fake_complete,
            checkpointer=first,
        )
        graph.invoke(_user_turn("hello"), _THREAD)
    finally:
        first.conn.close()

    second = sqlite_checkpointer(path)
    try:
        restored = build_graph(
            settings=settings,
            complete_fn=fake_complete,
            checkpointer=second,
        )
        snap = restored.get_state(_THREAD)
        messages = snap.values["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert [m["content"] for m in messages] == ["hello", "plain reply"]
    finally:
        second.conn.close()


def test_sqlite_get_state_during_invoke_is_serialized(
    settings: Settings, tmp_path: Path
):
    """Worker invoke and script-thread get_state share one locked SqliteSaver."""
    entered = threading.Event()
    release = threading.Event()
    read_ok = threading.Event()

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        entered.set()
        if not release.wait(timeout=2):
            raise AssertionError("get_state did not run during invoke")
        return "plain reply"

    saver = sqlite_checkpointer(tmp_path / "graph.sqlite")
    try:
        graph = build_graph(
            settings=settings,
            complete_fn=fake_complete,
            checkpointer=saver,
        )

        def reader() -> None:
            if not entered.wait(timeout=2):
                raise AssertionError("invoke never reached the model")
            graph.get_state(_THREAD)
            read_ok.set()
            release.set()

        worker = threading.Thread(
            target=graph.invoke, args=(_user_turn("hello"), _THREAD)
        )
        poller = threading.Thread(target=reader)
        worker.start()
        poller.start()
        worker.join(timeout=5)
        poller.join(timeout=5)
        assert not worker.is_alive()
        assert not poller.is_alive()
        assert read_ok.is_set()
        messages = graph.get_state(_THREAD).values["messages"]
        assert [m["content"] for m in messages] == ["hello", "plain reply"]
    finally:
        saver.conn.close()
