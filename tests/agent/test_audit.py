# test_audit.py

"""Tests for the append-only tool-use audit log. No dataset rows in the file."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.agent.audit import (
    DECISION_AUTO,
    OUTCOME_SUCCESS,
    append_tool_use,
    audit_log_path,
    code_text_for_audit,
    default_audit_dir,
    source_id_for_audit,
)
from src.agent.code_approval import TOOL_RUN_ANALYSIS_CODE
from src.agent.execute import run_allowlisted_tool
from src.agent.graph import build_graph
from src.agent.state import HITL_MODE_AUTO, empty_agent_state
from src.config import load_settings
from src.storage.registry import save_file_source
from src.tools.load_full_file import STATUS_NEEDS_APPROVAL
from src.validation.data_files import FileValidationError

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}


@pytest.fixture
def settings(tmp_path):
    """Frozen settings whose upload_dir is under tmp_path."""
    return load_settings(
        environ={
            "OLLAMA_HOST": "http://ollama.test:11434",
            **_PLACEHOLDER_MODELS,
        },
        load_dotenv_file=False,
        project_root=tmp_path,
    )


def _lines(settings) -> list[str]:
    path = audit_log_path(default_audit_dir(settings.upload_dir))
    if not path.is_file():
        return []
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_default_audit_dir_is_logs_sibling_of_upload_dir(settings):
    """Audit files go under upload_dir.parent/logs. There is no LOG_DIR env var."""
    assert default_audit_dir(settings.upload_dir) == (
        settings.upload_dir.parent / "logs"
    )


def test_allowlisted_tool_writes_one_audit_line(
    settings, tmp_path: Path, sample_sales_csv: Path
):
    """A successful tool run appends one JSONL record."""
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="sales.csv"
    )
    run_allowlisted_tool(
        "read_file_sample", {"source_id": saved.source_id}, settings
    )
    lines = _lines(settings)
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tool"] == "read_file_sample"
    assert record["source_id"] == saved.source_id
    datetime.fromisoformat(record["timestamp"])


def test_audit_line_omits_file_contents(
    settings, tmp_path: Path, sample_sales_csv: Path
):
    """The audit record is identities only, not rows or stored paths."""
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="sales.csv"
    )
    run_allowlisted_tool(
        "read_file_sample", {"source_id": saved.source_id}, settings
    )
    line = _lines(settings)[0]
    record = json.loads(line)
    assert set(record) == {"timestamp", "tool", "source_id"}
    assert "2024-01-01" not in line
    assert "North" not in line
    assert "rows" not in record
    if saved.stored_path is not None:
        assert str(saved.stored_path) not in line


def test_second_tool_use_appends(
    settings, tmp_path: Path, sample_sales_csv: Path
):
    """A later tool use adds a line. The first line stays."""
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    save_file_source(incoming, settings.upload_dir, original_name="sales.csv")
    run_allowlisted_tool("list_available_sources", {}, settings)
    run_allowlisted_tool("list_available_sources", {}, settings)
    lines = _lines(settings)
    assert len(lines) == 2
    assert json.loads(lines[0])["tool"] == "list_available_sources"
    assert json.loads(lines[1])["tool"] == "list_available_sources"


def test_list_sources_audit_has_null_source_id(settings):
    """A tool without source_id still writes the key, as null."""
    run_allowlisted_tool("list_available_sources", {}, settings)
    record = json.loads(_lines(settings)[0])
    assert record["source_id"] is None


def test_failed_tool_does_not_write_audit_line(settings):
    """A domain tool error is not recorded as a completed use."""
    with pytest.raises(FileValidationError, match="Unknown source_id"):
        run_allowlisted_tool(
            "read_file_sample", {"source_id": "file-missing"}, settings
        )
    assert _lines(settings) == []


def test_append_tool_use_writes_injected_timestamp(tmp_path: Path):
    """Tests may pin the timestamp. The writer does not invent extra keys."""
    when = datetime(2026, 9, 12, 16, 30, tzinfo=timezone.utc)
    record = append_tool_use(
        tmp_path / "logs",
        tool="profile_source",
        source_id="file-a",
        now=when,
    )
    assert record == {
        "timestamp": "2026-09-12T16:30:00+00:00",
        "tool": "profile_source",
        "source_id": "file-a",
    }


def test_append_tool_use_writes_code_decision_and_outcome(tmp_path: Path):
    """A generated-code record stores the text, HITL decision, and outcome."""
    when = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)
    record = append_tool_use(
        tmp_path / "logs",
        tool="run_analysis_code",
        source_id="file-a",
        now=when,
        code="print(1)",
        decision=DECISION_AUTO,
        outcome=OUTCOME_SUCCESS,
    )
    assert record == {
        "timestamp": "2026-09-13T10:00:00+00:00",
        "tool": "run_analysis_code",
        "source_id": "file-a",
        "code": "print(1)",
        "decision": DECISION_AUTO,
        "outcome": OUTCOME_SUCCESS,
        "code_truncated": False,
    }
    assert "rows" not in record
    assert "stdout" not in record


def test_code_text_for_audit_is_capped():
    """Over-long generated text is truncated. Dataset rows are not added."""
    text, truncated = code_text_for_audit(
        TOOL_RUN_ANALYSIS_CODE,
        {"source_id": "file-a", "code": "print(1)" + "x" * 50},
        limit=8,
    )
    assert text == "print(1)"
    assert truncated is True
    assert "rows" not in text


def test_code_text_for_audit_fits_without_truncation():
    """A short snippet is stored in full and not flagged as cut."""
    text, truncated = code_text_for_audit(
        TOOL_RUN_ANALYSIS_CODE,
        {"source_id": "file-a", "code": "print(1)"},
        limit=80,
    )
    assert text == "print(1)"
    assert truncated is False


def test_code_text_for_audit_limit_below_one_does_not_truncate():
    """limit < 1 is no cap, so a long snippet is not flagged as cut."""
    raw = "print(1)" + "x" * 50
    text, truncated = code_text_for_audit(
        TOOL_RUN_ANALYSIS_CODE,
        {"source_id": "file-a", "code": raw},
        limit=0,
    )
    assert text == raw
    assert truncated is False


def test_append_tool_use_marks_truncated_code(tmp_path: Path):
    """A cut snippet is flagged so a reviewer does not treat it as the full text."""
    record = append_tool_use(
        tmp_path / "logs",
        tool="run_analysis_code",
        source_id="file-a",
        code="print(1)",
        code_truncated=True,
        decision=DECISION_AUTO,
        outcome=OUTCOME_SUCCESS,
    )
    assert record["code"] == "print(1)"
    assert record["code_truncated"] is True


def test_source_id_for_audit_does_not_invent_an_id():
    """Missing or blank source_id stays unset."""
    assert source_id_for_audit({}) is None
    assert source_id_for_audit({"source_id": "   "}) is None
    assert source_id_for_audit({"source_id": "file-a"}) == "file-a"
    assert source_id_for_audit({"connection_id": "postgres-a"}) == "postgres-a"


def test_append_tool_use_write_failure_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A disk error must not turn a finished tool into a failed turn."""
    caplog.set_level(logging.WARNING, logger="src.agent.audit")

    def boom(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", boom)
    record = append_tool_use(
        tmp_path / "logs",
        tool="list_available_sources",
        source_id=None,
    )
    assert record["tool"] == "list_available_sources"
    assert "audit write failed" in caplog.text
    assert "OSError" in caplog.text
    assert not (tmp_path / "logs" / "audit.jsonl").is_file()


def test_needs_approval_does_not_write_audit_line(
    tmp_path: Path, sample_sales_csv: Path
):
    """A paused over-limit load is not recorded as a completed use."""
    settings = load_settings(
        environ={
            "OLLAMA_HOST": "http://ollama.test:11434",
            "MAX_FULL_LOAD_ROWS": "1",
            **_PLACEHOLDER_MODELS,
        },
        load_dotenv_file=False,
        project_root=tmp_path,
    )
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="sales.csv"
    )
    result = run_allowlisted_tool(
        "load_full_file", {"source_id": saved.source_id}, settings
    )
    assert result["status"] == STATUS_NEEDS_APPROVAL
    assert _lines(settings) == []


def test_full_fake_run_writes_expected_audit_keys(
    tmp_path: Path, sample_sales_csv: Path
):
    """One Auto turn records profile, tools, generated code, HITL, and sandbox outcome."""
    settings = load_settings(
        environ={"OLLAMA_HOST": "http://ollama.test:11434", **_PLACEHOLDER_MODELS},
        load_dotenv_file=False,
        project_root=tmp_path,
    )
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="sales.csv"
    )
    replies = [
        json.dumps({"name": "list_available_sources", "arguments": {}}),
        json.dumps(
            {
                "name": "read_file_sample",
                "arguments": {"source_id": saved.source_id},
            }
        ),
        json.dumps(
            {
                "name": "run_analysis_code",
                "arguments": {
                    "source_id": saved.source_id,
                    "code": "print(1)\n",
                },
            }
        ),
        "done",
    ]

    def fake_complete(prompt: str, **kwargs: object) -> str:
        return replies.pop(0)

    graph = build_graph(settings=settings, complete_fn=fake_complete)
    state = empty_agent_state(hitl_mode=HITL_MODE_AUTO)
    state["messages"] = [{"role": "user", "content": "profile then sample then print"}]
    result = graph.invoke(state, {"configurable": {"thread_id": "audit-full-run"}})
    assert result["error"] is None
    records = [json.loads(line) for line in _lines(settings)]
    assert [row["tool"] for row in records] == [
        "profile_source",
        "list_available_sources",
        "read_file_sample",
        "run_analysis_code",
    ]
    identity_keys = {"timestamp", "tool", "source_id"}
    for row in records[:3]:
        assert set(row) == identity_keys
        datetime.fromisoformat(row["timestamp"])
    assert records[0]["source_id"] == saved.source_id
    assert records[1]["source_id"] is None
    assert records[2]["source_id"] == saved.source_id
    code_row = records[3]
    assert set(code_row) == identity_keys | {
        "code",
        "decision",
        "outcome",
        "code_truncated",
    }
    assert code_row["code"] == "print(1)"
    assert code_row["decision"] == DECISION_AUTO
    assert code_row["outcome"] == OUTCOME_SUCCESS
    assert code_row["code_truncated"] is False
    assert "rows" not in json.dumps(records)
    assert "North" not in json.dumps(records)
