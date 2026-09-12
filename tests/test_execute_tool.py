# test_execute_tool.py

"""Tests for allowlisted tool dispatch and tool-call JSON parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.execute import parse_tool_call, run_allowlisted_tool
from src.agent.json_output import JsonSchemaError
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
    """Settings whose upload_dir is under tmp_path."""
    return load_settings(
        environ={
            "OLLAMA_HOST": "http://ollama.test:11434",
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            **_PLACEHOLDER_MODELS,
        },
        load_dotenv_file=False,
        project_root=tmp_path,
    )


def test_parse_plain_text_is_not_a_tool_call():
    """Prose is a user-facing reply, not a guessed tool name."""
    assert parse_tool_call("hello") is None


def test_parse_json_without_name_is_not_a_tool_call():
    """A JSON object that is not a tool call stays a text reply."""
    assert parse_tool_call('{"content": "hello"}') is None


def test_parse_tool_call_reads_name_and_arguments():
    """A registered-shape object is a tool call."""
    assert parse_tool_call(
        '{"name": "list_available_sources", "arguments": {}}'
    ) == {"name": "list_available_sources", "arguments": {}}


def test_parse_tool_call_strips_fences():
    """A fenced tool JSON is still a tool call."""
    raw = '```json\n{"name": "list_available_sources"}\n```'
    assert parse_tool_call(raw) == {
        "name": "list_available_sources",
        "arguments": {},
    }


def test_parse_tool_call_rejects_extra_keys():
    """Extra keys are a schema miss, not a guessed call."""
    with pytest.raises(JsonSchemaError, match="Unexpected keys"):
        parse_tool_call('{"name": "list_available_sources", "extra": 1}')


def test_unknown_tool_is_rejected(settings):
    """A name outside the registry is not run."""
    with pytest.raises(ValueError, match="Unknown tool"):
        run_allowlisted_tool("not_a_tool", {}, settings)


def test_llm_cannot_override_upload_dir(
    settings, tmp_path: Path, sample_sales_csv: Path
):
    """upload_dir from the model is ignored; settings win."""
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="sales.csv"
    )
    result = run_allowlisted_tool(
        "list_available_sources",
        {"upload_dir": str(tmp_path / "evil")},
        settings,
    )
    assert result["sources"][0]["source_id"] == saved.source_id


def test_llm_cannot_sample_by_path(
    settings, tmp_path: Path, sample_sales_csv: Path
):
    """A raw path from the model is not a way around the source registry."""
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="sales.csv"
    )
    assert saved.stored_path is not None
    with pytest.raises(FileValidationError, match="source_id"):
        run_allowlisted_tool(
            "read_file_sample",
            {"path": str(saved.stored_path)},
            settings,
        )


def test_llm_path_is_ignored_when_source_id_is_present(
    settings, tmp_path: Path, sample_sales_csv: Path
):
    """source_id wins; a second file path from the model is not read."""
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="sales.csv"
    )
    other = settings.upload_dir / "other.csv"
    other.write_text("sku,qty\nA,1\n", encoding="utf-8")
    result = run_allowlisted_tool(
        "read_file_sample",
        {"source_id": saved.source_id, "path": str(other)},
        settings,
    )
    assert result["columns"] == ["date", "region", "revenue"]
    assert "sku" not in result["columns"]
