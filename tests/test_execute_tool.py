# test_execute_tool.py

"""Tests for allowlisted tool dispatch and tool-call JSON parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.execute import (
    ToolValidationError,
    _validate_arg_types,
    interpret_model_reply,
    parse_tool_call,
    retry_prompt_after_validation,
    run_allowlisted_tool,
    validate_tool_call,
)
from src.agent.json_output import STRICT_RETRY_INSTRUCTION, JsonSchemaError
from src.config import load_settings
from src.storage.registry import save_file_source

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
    with pytest.raises(ToolValidationError, match="Unknown tool"):
        run_allowlisted_tool("not_a_tool", {}, settings)


def test_validate_tool_call_accepts_list_sources():
    """An empty argument object is valid for list_available_sources."""
    assert validate_tool_call("list_available_sources", {}) == {
        "name": "list_available_sources",
        "arguments": {},
    }


def test_validate_tool_call_strips_app_keys():
    """upload_dir from the model is dropped before schema check."""
    checked = validate_tool_call(
        "list_available_sources", {"upload_dir": "/evil"}
    )
    assert checked["arguments"] == {}


def test_validate_missing_source_id_is_rejected():
    """read_file_sample requires source_id. Fields are not guessed."""
    with pytest.raises(ToolValidationError, match="Missing keys: source_id"):
        validate_tool_call("read_file_sample", {})


def test_validate_path_argument_is_rejected():
    """path is not in the public schema, even with a source_id."""
    with pytest.raises(ToolValidationError, match="Unexpected keys: path"):
        validate_tool_call(
            "read_file_sample",
            {"source_id": "file-abc", "path": "sales.csv"},
        )


def test_validate_n_rows_must_be_a_positive_integer():
    """A non-integer n_rows is a schema miss, not coerced."""
    with pytest.raises(ToolValidationError, match="n_rows must be an integer"):
        validate_tool_call(
            "read_file_sample",
            {"source_id": "file-abc", "n_rows": "10"},
        )


def test_validate_n_rows_rejects_boolean():
    """True is not an integer n_rows. isinstance(True, int) is not enough."""
    with pytest.raises(ToolValidationError, match="n_rows must be an integer"):
        validate_tool_call(
            "read_file_sample",
            {"source_id": "file-abc", "n_rows": True},
        )


def test_boolean_arg_rejects_non_bool():
    """A boolean property is not coerced from a string."""
    with pytest.raises(JsonSchemaError, match="flag must be a boolean"):
        _validate_arg_types({"flag": "yes"}, {"properties": {"flag": {"type": "boolean"}}})


def test_number_arg_rejects_non_number():
    """A number property is not coerced from a string."""
    with pytest.raises(JsonSchemaError, match="score must be a number"):
        _validate_arg_types(
            {"score": "1.5"}, {"properties": {"score": {"type": "number"}}}
        )


def test_array_arg_rejects_non_list():
    """An array property is not coerced from a string."""
    with pytest.raises(JsonSchemaError, match="ids must be an array"):
        _validate_arg_types({"ids": "a"}, {"properties": {"ids": {"type": "array"}}})


def test_unsupported_arg_type_is_rejected():
    """An undeclared schema type is not skipped."""
    with pytest.raises(JsonSchemaError, match="Unsupported argument type: object"):
        _validate_arg_types({"meta": {}}, {"properties": {"meta": {"type": "object"}}})


def test_interpret_plain_text_is_not_a_tool_call():
    """A user-facing reply is not validated as a tool."""
    assert interpret_model_reply("There is 1 source.") is None


def test_interpret_unknown_tool_raises():
    """A named but unregistered tool is a validation error."""
    with pytest.raises(ToolValidationError, match="Unknown tool"):
        interpret_model_reply('{"name": "not_a_tool", "arguments": {}}')


def test_retry_prompt_includes_strict_instruction_and_error():
    """The retry prompt uses the 3.2 instruction and names the validation miss."""
    text = retry_prompt_after_validation("base", "Unknown tool: x.")
    assert text.startswith("base")
    assert STRICT_RETRY_INSTRUCTION in text
    assert "Unknown tool: x." in text


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


def test_llm_cannot_sample_by_path(settings):
    """A raw path from the model is rejected before the handler runs."""
    with pytest.raises(ToolValidationError, match="Missing keys: source_id"):
        run_allowlisted_tool(
            "read_file_sample",
            {"path": str(settings.upload_dir / "sales.csv")},
            settings,
        )
