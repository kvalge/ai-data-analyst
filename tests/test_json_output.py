# test_json_output.py

"""Tests for fence stripping, JSON parse, schema checks, and retry policy."""

import pytest

from src.agent.json_output import (
    FAIL,
    FIRST_PARSE_FAILURE,
    RETRY_STRICT,
    STRICT_RETRY_INSTRUCTION,
    JsonParseError,
    JsonSchemaError,
    decide_after_parse_failure,
    parse_json_output,
    strip_markdown_fences,
)

_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


def test_strip_json_fence():
    """A ```json wrapper is removed before parse."""
    raw = '```json\n{"name": "list_available_sources"}\n```'
    assert strip_markdown_fences(raw) == '{"name": "list_available_sources"}'


def test_strip_plain_fence():
    """A language-less ``` wrapper is removed too."""
    raw = '```\n{"name": "list_available_sources"}\n```'
    assert strip_markdown_fences(raw) == '{"name": "list_available_sources"}'


def test_parse_fenced_json():
    """Fenced JSON loads and matches the schema."""
    raw = '```json\n{"name": "list_available_sources"}\n```'
    assert parse_json_output(raw, _SCHEMA) == {"name": "list_available_sources"}


def test_parse_bare_json():
    """JSON without fences still loads."""
    assert parse_json_output('{"name": "profile_source"}', _SCHEMA) == {
        "name": "profile_source"
    }


def test_prose_around_json_is_not_extracted():
    """JSON buried in commentary is not guessed out."""
    with pytest.raises(JsonParseError, match="Could not parse JSON"):
        parse_json_output('Here you go:\n{"name": "x"}\nThanks', _SCHEMA)


def test_fence_plus_trailing_prose_is_not_extracted():
    """A fence followed by extra text is not trimmed down to the object."""
    with pytest.raises(JsonParseError, match="Could not parse JSON"):
        parse_json_output('```json\n{"name": "x"}\n```\nThanks', _SCHEMA)


def test_bad_json_raises_parse_error():
    """Malformed text is a parse error, not a guessed object."""
    with pytest.raises(JsonParseError, match="Could not parse JSON"):
        parse_json_output("not-json", _SCHEMA)


def test_fenced_bad_json_raises_parse_error():
    """A fence around invalid JSON is still a parse error."""
    with pytest.raises(JsonParseError, match="Could not parse JSON"):
        parse_json_output("```json\nnot-json\n```", _SCHEMA)


def test_schema_miss_missing_required_key():
    """A valid object with a missing required key is not filled in."""
    with pytest.raises(JsonSchemaError, match="Missing keys: name"):
        parse_json_output("{}", _SCHEMA)


def test_schema_miss_unexpected_key():
    """Extra keys are rejected when additionalProperties is false."""
    with pytest.raises(JsonSchemaError, match="Unexpected keys: extra"):
        parse_json_output('{"name": "x", "extra": 1}', _SCHEMA)


def test_extra_keys_allowed_when_additional_properties_omitted():
    """Omitting additionalProperties follows JSON Schema and allows extra keys."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    assert parse_json_output('{"name": "x", "extra": 1}', schema) == {
        "name": "x",
        "extra": 1,
    }


def test_schema_miss_non_object():
    """A JSON array is not coerced into an object."""
    with pytest.raises(JsonSchemaError, match="Expected a JSON object"):
        parse_json_output("[]", _SCHEMA)


def test_retry_once_after_first_failure():
    """The first parse failure may retry with a stricter instruction."""
    assert decide_after_parse_failure(FIRST_PARSE_FAILURE) == RETRY_STRICT


def test_strict_retry_instruction_forbids_fences():
    """The retry prompt asks for a bare JSON object."""
    assert "JSON object only" in STRICT_RETRY_INSTRUCTION


def test_fail_after_second_failure():
    """A second parse failure is a visible fail, not another guess."""
    assert decide_after_parse_failure(2) == FAIL


def test_fail_after_later_failures():
    """There is no third attempt."""
    assert decide_after_parse_failure(3) == FAIL


def test_zero_failures_is_not_a_retry_decision():
    """The policy is only called after a failure."""
    with pytest.raises(ValueError, match="failure_count"):
        decide_after_parse_failure(0)
