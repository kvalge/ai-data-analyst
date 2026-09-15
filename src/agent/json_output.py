# json_output.py

"""Parse LLM JSON: strip fences, load, validate. One stricter retry, then fail."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from src.config import bound_text

_LOG = logging.getLogger(__name__)

RETRY_STRICT = "retry_strict"
FAIL = "fail"
FIRST_PARSE_FAILURE = 1
STRICT_RETRY_INSTRUCTION = (
    "Reply with a single JSON object only. No markdown fences, no commentary."
)
_RETRY_SEP = "\n\n"

_FENCED = re.compile(
    r"^```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)[ \t]*\r?\n?```$",
    flags=re.IGNORECASE | re.DOTALL,
)
_UNBALANCED_FENCE_LEAD = re.compile(r"^```[A-Za-z0-9_+-]*[ \t]*\r?\n")
_UNBALANCED_FENCE_TAIL = re.compile(r"\r?\n?[ \t]*```[ \t]*$")


class JsonOutputError(ValueError):
    """LLM text was not valid JSON for the given schema."""


class JsonParseError(JsonOutputError):
    """json.loads failed after fence stripping."""


class JsonSchemaError(JsonOutputError):
    """Parsed JSON does not match the schema. Fields are not invented."""


def retry_prompt_after_validation(
    base_prompt: str,
    error: str,
    *,
    limit: int | None = None,
    instruction: str = STRICT_RETRY_INSTRUCTION,
) -> str:
    """Put the retry instruction first so prompt truncation cannot drop it."""
    head_parts = [instruction] if instruction else []
    head_parts.append(f"Previous model output failed validation: {error}")
    head = _RETRY_SEP.join(head_parts)
    if limit is None or limit < 1:
        return _RETRY_SEP.join((head, base_prompt))
    room = limit - len(head) - len(_RETRY_SEP)
    if room < 1:
        return bound_text(head, limit)
    return _RETRY_SEP.join((head, bound_text(base_prompt, room)))


def strip_markdown_fences(text: str) -> str:
    """Remove a wrapping markdown fence (optional language tag).

    A model sometimes emits only half of the fence. One unbalanced delimiter
    at the start or end is a wrapping artifact, so it is removed too. Two or
    more are left alone: that is content or prose, and prose is never
    trimmed down to the object it surrounds.
    """
    stripped = text.strip()
    match = _FENCED.fullmatch(stripped)
    if match:
        return match.group(1).strip()
    if stripped.count("```") == 1:
        stripped = _UNBALANCED_FENCE_LEAD.sub("", stripped, count=1)
        stripped = _UNBALANCED_FENCE_TAIL.sub("", stripped, count=1)
    return stripped.strip()


def parse_json_output(text: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Strip fences, json.loads, then validate. Does not fill missing keys."""
    stripped = strip_markdown_fences(text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        _LOG.info("json output parse failed")
        raise JsonParseError("Could not parse JSON.") from exc
    validate_json_object(payload, schema)
    return payload


def validate_json_object(payload: object, schema: Mapping[str, Any]) -> None:
    """Require a dict that matches required keys and additionalProperties."""
    if schema.get("type", "object") != "object":
        raise JsonSchemaError("Only object schemas are supported.")
    if not isinstance(payload, dict):
        raise JsonSchemaError("Expected a JSON object.")
    required = schema.get("required", [])
    if not isinstance(required, list):
        raise JsonSchemaError("Schema required list is invalid.")
    missing = [str(key) for key in required if key not in payload]
    if missing:
        raise JsonSchemaError("Missing keys: " + ", ".join(missing))
    if schema.get("additionalProperties") is False:
        allowed = set(schema.get("properties", {}))
        extra = sorted(str(key) for key in payload if key not in allowed)
        if extra:
            raise JsonSchemaError("Unexpected keys: " + ", ".join(extra))


def decide_after_parse_failure(failure_count: int) -> str:
    """After the first failure, retry once with a stricter instruction. Then fail."""
    if failure_count < FIRST_PARSE_FAILURE:
        raise ValueError("failure_count must be at least 1.")
    if failure_count == FIRST_PARSE_FAILURE:
        return RETRY_STRICT
    return FAIL
