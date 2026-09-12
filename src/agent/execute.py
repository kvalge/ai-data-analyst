# execute.py

"""Run allowlisted Phase 3 tools. App paths and limits come from settings."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from src.agent.json_output import (
    STRICT_RETRY_INSTRUCTION,
    JsonSchemaError,
    strip_markdown_fences,
    validate_json_object,
)
from src.config import Settings
from src.tools.registry import TOOL_REGISTRY

_LOG = logging.getLogger(__name__)

# The LLM must not supply these. Paths and limits come from Settings.
_APP_ARG_KEYS = frozenset(
    {
        "upload_dir",
        "cache_dir",
        "settings",
        "max_bytes",
        "max_full_load_rows",
        "include_postgres",
    }
)

TOOL_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "arguments": {"type": "object"},
    },
    "required": ["name"],
    "additionalProperties": False,
}


class ToolValidationError(ValueError):
    """Tool name or arguments are invalid. Do not guess a call."""


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Return {name, arguments} if `text` is a tool JSON object. Else None.

    Does not extract JSON from surrounding prose.
    """
    stripped = strip_markdown_fences(text)
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "name" not in payload:
        return None
    validate_json_object(payload, TOOL_CALL_SCHEMA)
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise JsonSchemaError("Tool arguments must be an object.")
    return {"name": str(payload["name"]), "arguments": arguments}


def validate_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Allowlist `name` and match remaining arguments to the tool schema."""
    if name not in TOOL_REGISTRY:
        raise ToolValidationError(f"Unknown tool: {name}.")
    schema = TOOL_REGISTRY[name].contract.input_schema
    llm_args = {
        key: value for key, value in arguments.items() if key not in _APP_ARG_KEYS
    }
    try:
        validate_json_object(llm_args, schema)
        _validate_arg_types(llm_args, schema)
    except JsonSchemaError as exc:
        raise ToolValidationError(str(exc)) from exc
    return {"name": name, "arguments": llm_args}


def interpret_model_reply(text: str) -> dict[str, Any] | None:
    """Return a validated tool call, or None if `text` is a user-facing reply."""
    call = parse_tool_call(text)
    if call is None:
        return None
    return validate_tool_call(call["name"], call["arguments"])


def retry_prompt_after_validation(base_prompt: str, error: str) -> str:
    """Append the 3.2 strict instruction and the validation error. Do not guess."""
    return "\n\n".join(
        (
            base_prompt,
            STRICT_RETRY_INSTRUCTION,
            f"Previous tool call failed validation: {error}",
        )
    )


def run_allowlisted_tool(
    name: str,
    arguments: dict[str, Any],
    settings: Settings,
    *,
    include_postgres: bool = False,
) -> dict[str, Any]:
    """Look up `name` on the registry, inject app args, and run the handler."""
    checked = validate_tool_call(name, arguments)
    kwargs = _injected_kwargs(
        checked["name"], settings, checked["arguments"], include_postgres
    )
    _LOG.info("execute tool=%s", checked["name"])
    return TOOL_REGISTRY[checked["name"]].handler(**kwargs)


def _validate_arg_types(payload: dict[str, Any], schema: Mapping[str, Any]) -> None:
    """Check declared JSON Schema types. Missing optional keys are skipped."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for key, spec in properties.items():
        if key not in payload or not isinstance(spec, dict):
            continue
        expected = spec.get("type")
        if expected is None:
            continue
        value = payload[key]
        if expected == "string":
            if not isinstance(value, str):
                raise JsonSchemaError(f"{key} must be a string.")
            continue
        if expected == "integer":
            if type(value) is not int:
                raise JsonSchemaError(f"{key} must be an integer.")
            minimum = spec.get("minimum")
            if isinstance(minimum, int) and value < minimum:
                raise JsonSchemaError(f"{key} must be at least {minimum}.")
            continue
        if expected == "number":
            if type(value) not in (int, float):
                raise JsonSchemaError(f"{key} must be a number.")
            continue
        if expected == "boolean":
            if type(value) is not bool:
                raise JsonSchemaError(f"{key} must be a boolean.")
            continue
        if expected == "array":
            if not isinstance(value, list):
                raise JsonSchemaError(f"{key} must be an array.")
            continue
        raise JsonSchemaError(f"Unsupported argument type: {expected}.")


def _injected_kwargs(
    name: str,
    settings: Settings,
    args: dict[str, Any],
    include_postgres: bool,
) -> dict[str, Any]:
    """Build handler kwargs. Limits and paths always come from settings."""
    if name == "list_available_sources":
        return {
            "upload_dir": settings.upload_dir,
            "settings": settings,
            "include_postgres": include_postgres,
        }
    if name == "read_file_sample":
        n_rows = args.get("n_rows", settings.sample_n_rows)
        return {
            "upload_dir": settings.upload_dir,
            "n_rows": int(n_rows),
            "max_bytes": settings.max_upload_bytes,
            "source_id": args.get("source_id"),
            "path": None,
        }
    if name == "profile_source":
        n_rows = args.get("n_rows", settings.sample_n_rows)
        return {
            "upload_dir": settings.upload_dir,
            "cache_dir": settings.cache_dir,
            "n_rows": int(n_rows),
            "max_bytes": settings.max_upload_bytes,
            "source_id": args["source_id"],
            "settings": settings,
        }
    if name == "load_full_file":
        return {
            "upload_dir": settings.upload_dir,
            "source_id": args["source_id"],
            "max_full_load_rows": settings.max_full_load_rows,
            "max_bytes": settings.max_upload_bytes,
            "settings": settings,
        }
    raise ValueError(f"Unknown tool: {name}.")
