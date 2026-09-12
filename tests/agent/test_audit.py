# test_audit.py

"""Tests for the append-only tool-use audit log. No dataset rows in the file."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.agent.audit import (
    append_tool_use,
    audit_log_path,
    default_audit_dir,
    source_id_for_audit,
)
from src.agent.execute import run_allowlisted_tool
from src.config import load_settings
from src.storage.registry import save_file_source
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


def test_source_id_for_audit_does_not_invent_an_id():
    """Missing or blank source_id stays unset."""
    assert source_id_for_audit({}) is None
    assert source_id_for_audit({"source_id": "   "}) is None
    assert source_id_for_audit({"source_id": "file-a"}) == "file-a"
