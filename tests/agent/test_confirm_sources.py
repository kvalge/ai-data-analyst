# test_confirm_sources.py

"""Tests for source confirmation decisions and the graph interrupt payload."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

from src.agent.confirm_sources import (
    ACTION_ABORT,
    ACTION_CONFIRM,
    ACTION_SELECT,
    KIND_CONFIRM_SOURCES,
    REASON_EMPTY_SCHEMA,
    REASON_MULTIPLE_SOURCES,
    REASON_NO_SOURCES,
    apply_confirm_decision,
    build_interrupt_payload,
    decide_confirm_reason,
    sample_is_empty,
)
from src.agent.graph import build_graph
from src.agent.state import empty_agent_state
from src.config import load_settings
from src.storage.registry import save_file_source
from src.ui.source_confirm import reason_message

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}

_THREAD = {"configurable": {"thread_id": "confirm-thread"}}


@pytest.fixture
def settings(tmp_path):
    """Settings with an empty upload dir."""
    return load_settings(
        environ={"OLLAMA_HOST": "http://ollama.test:11434", **_PLACEHOLDER_MODELS},
        load_dotenv_file=False,
        project_root=tmp_path,
    )


def _interrupt_value(result: dict[str, Any]) -> dict[str, Any]:
    items = result.get("__interrupt__") or []
    assert items, "expected a confirm_sources interrupt"
    value = items[0].value
    assert isinstance(value, dict)
    return value


def _user_turn(text: str):
    state = empty_agent_state()
    state["messages"] = [{"role": "user", "content": text}]
    return state


def test_decide_no_sources_interrupts():
    """Zero registered sources cannot proceed."""
    reason, selected = decide_confirm_reason([], [])
    assert reason == REASON_NO_SOURCES
    assert selected == []


def test_decide_multiple_sources_without_selection_interrupts():
    """Two sources and no selection is a pause, not a guessed id."""
    sources = [
        {"source_id": "file-a", "kind": "file", "original_name": "a.csv"},
        {"source_id": "file-b", "kind": "file", "original_name": "b.csv"},
    ]
    reason, selected = decide_confirm_reason(sources, [])
    assert reason == REASON_MULTIPLE_SOURCES
    assert selected == []


def test_decide_one_source_auto_selects():
    """A single registered source is selected without an interrupt reason."""
    sources = [{"source_id": "file-a", "kind": "file", "original_name": "a.csv"}]
    reason, selected = decide_confirm_reason(sources, [])
    assert reason is None
    assert selected == ["file-a"]


def test_sample_is_empty_when_no_rows():
    """A header-only sample is empty even when columns exist."""
    assert sample_is_empty({"columns": ["date", "region", "revenue"], "row_count": 0})


def test_payload_omits_dataset_rows():
    """The interrupt value lists identities, not records."""
    payload = build_interrupt_payload(
        REASON_MULTIPLE_SOURCES,
        [
            {
                "source_id": "file-a",
                "kind": "file",
                "original_name": "sales.csv",
                "stored_path": "/tmp/sales.csv",
                "sha256": "abc",
            }
        ],
        [],
    )
    assert payload["kind"] == KIND_CONFIRM_SOURCES
    assert payload["sources"] == [
        {
            "source_id": "file-a",
            "kind": "file",
            "original_name": "sales.csv",
        }
    ]
    assert "stored_path" not in payload["sources"][0]
    assert "date,region,revenue" not in str(payload)


def test_abort_clears_selection():
    """Abort is a visible stop, not a guessed source."""
    updates = apply_confirm_decision({"action": ACTION_ABORT}, [], [])
    assert updates["error"] == "Source confirmation was aborted."
    assert updates["source_ids"] == []


def test_confirm_records_cleared_empty_source_id():
    """Confirm remembers the id so an empty schema is not re-prompted."""
    sources = [{"source_id": "file-a", "kind": "file", "original_name": "a.csv"}]
    prior = ["file-old"]
    updates = apply_confirm_decision(
        {"action": ACTION_CONFIRM, "source_ids": ["file-a"]},
        sources,
        [],
        cleared_empty_source_ids=prior,
    )
    assert updates["cleared_empty_source_ids"] == ["file-old", "file-a"]
    assert prior == ["file-old"]


def test_confirm_does_not_duplicate_a_cleared_source_id():
    """Confirming the same id again must not grow the checkpointed list."""
    sources = [{"source_id": "file-a", "kind": "file", "original_name": "a.csv"}]
    updates = apply_confirm_decision(
        {"action": ACTION_CONFIRM, "source_ids": ["file-a"]},
        sources,
        [],
        cleared_empty_source_ids=["file-a"],
    )
    assert updates["cleared_empty_source_ids"] == ["file-a"]


def test_select_rejects_unknown_id():
    """A source_id that is not registered is not accepted."""
    sources = [{"source_id": "file-a", "kind": "file", "original_name": "a.csv"}]
    updates = apply_confirm_decision(
        {"action": ACTION_SELECT, "source_ids": ["file-missing"]},
        sources,
        [],
    )
    assert updates["error"] == "Select a registered source."


def test_reason_message_covers_no_sources():
    """The UI copy for no sources does not invent a file."""
    assert "Upload" in reason_message(REASON_NO_SOURCES)


def test_graph_interrupts_when_no_sources(settings):
    """A turn with an empty registry pauses before the LLM."""
    called = False

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        nonlocal called
        called = True
        return "should not run"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(_user_turn("hello"), _THREAD)
    payload = _interrupt_value(result)
    assert called is False
    assert payload["reason"] == REASON_NO_SOURCES
    assert payload["kind"] == KIND_CONFIRM_SOURCES
    assert payload["sources"] == []


def test_graph_interrupts_when_multiple_sources_unselected(
    settings, tmp_path, sample_sales_csv: Path
):
    """Two files and no source_ids pause with both identities."""
    first = tmp_path / "sales.csv"
    first.write_bytes(sample_sales_csv.read_bytes())
    save_file_source(first, settings.upload_dir, original_name="sales.csv")
    second = tmp_path / "other.csv"
    second.write_text("date,region,revenue\n2024-01-02,west,2\n", encoding="utf-8")
    save_file_source(second, settings.upload_dir, original_name="other.csv")

    graph = build_graph(settings=settings, complete_fn=lambda *a, **k: "no")
    result = graph.invoke(_user_turn("hello"), _THREAD)
    payload = _interrupt_value(result)
    assert payload["reason"] == REASON_MULTIPLE_SOURCES
    names = {row["original_name"] for row in payload["sources"]}
    assert names == {"sales.csv", "other.csv"}
    assert "west" not in str(payload)


def test_graph_interrupts_empty_schema_when_source_already_selected(
    settings, tmp_path
):
    """Sidebar-style source_ids still pause on an empty sample."""
    incoming = tmp_path / "empty.csv"
    incoming.write_text("date,region,revenue\n", encoding="utf-8")
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="empty.csv"
    )
    state = _user_turn("hello")
    state["source_ids"] = [saved.source_id]
    graph = build_graph(settings=settings, complete_fn=lambda *a, **k: "no")
    payload = _interrupt_value(graph.invoke(state, _THREAD))
    assert payload["reason"] == REASON_EMPTY_SCHEMA
    assert payload["source_ids"] == [saved.source_id]


def test_graph_interrupts_when_sample_is_empty(settings, tmp_path):
    """A header-only file pauses as an empty schema."""
    incoming = tmp_path / "empty.csv"
    incoming.write_text("date,region,revenue\n", encoding="utf-8")
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="empty.csv"
    )
    graph = build_graph(settings=settings, complete_fn=lambda *a, **k: "no")
    result = graph.invoke(_user_turn("hello"), _THREAD)
    payload = _interrupt_value(result)
    assert payload["reason"] == REASON_EMPTY_SCHEMA
    assert payload["source_ids"] == [saved.source_id]
    assert payload["row_count"] == 0
    assert "date,region,revenue" not in str(payload["sources"])


def test_graph_resume_abort_does_not_call_llm(settings):
    """Abort ends the turn with a visible error and no assistant guess."""
    called = False

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        nonlocal called
        called = True
        return "should not run"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("hello"), _THREAD)
    result = graph.invoke(Command(resume={"action": ACTION_ABORT}), _THREAD)
    assert called is False
    assert result["error"] == "Source confirmation was aborted."
    assert [m["role"] for m in result["messages"]] == ["user"]
    assert "__interrupt__" not in result


def test_graph_skips_interrupt_when_source_already_selected(
    settings, tmp_path, sample_sales_csv: Path
):
    """A valid selected file does not pause the turn."""
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="sales.csv"
    )
    state = _user_turn("hello")
    state["source_ids"] = [saved.source_id]

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return "plain reply"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    result = graph.invoke(state, _THREAD)
    assert result.get("__interrupt__") is None
    assert result["source_ids"] == [saved.source_id]
    assert result["messages"][-1]["content"] == "plain reply"


def test_graph_confirm_empty_schema_then_replies(settings, tmp_path):
    """Confirming an empty sample proceeds; the model does not invent rows."""
    incoming = tmp_path / "empty.csv"
    incoming.write_text("date,region,revenue\n", encoding="utf-8")
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="empty.csv"
    )

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return "plain reply"

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("hello"), _THREAD)
    result = graph.invoke(
        Command(
            resume={
                "action": ACTION_CONFIRM,
                "source_ids": [saved.source_id],
            }
        ),
        _THREAD,
    )
    assert result.get("__interrupt__") is None
    assert result["source_ids"] == [saved.source_id]
    assert result["messages"][-1]["content"] == "plain reply"


def test_graph_does_not_reinterrupt_cleared_empty_source(settings, tmp_path):
    """A later turn with the same empty source does not pause again."""
    incoming = tmp_path / "empty.csv"
    incoming.write_text("date,region,revenue\n", encoding="utf-8")
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="empty.csv"
    )
    replies = ["first-reply", "second-reply"]

    def fake_complete(prompt: str, **kwargs: Any) -> str:
        return replies.pop(0)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    graph.invoke(_user_turn("hello"), _THREAD)
    graph.invoke(
        Command(
            resume={
                "action": ACTION_CONFIRM,
                "source_ids": [saved.source_id],
            }
        ),
        _THREAD,
    )
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "again"}],
            "source_ids": [saved.source_id],
        },
        _THREAD,
    )
    assert result.get("__interrupt__") is None
    assert result["messages"][-1]["content"] == "second-reply"
