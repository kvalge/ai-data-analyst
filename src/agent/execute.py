# execute.py

"""Run allowlisted Phase 3 tools. App paths and limits come from settings."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.json_output import (
    JsonSchemaError,
    strip_markdown_fences,
    validate_json_object,
)
from src.config import Settings
from src.tools.registry import TOOL_REGISTRY

_LOG = logging.getLogger(__name__)

# The LLM must not supply these. Paths and limits come from Settings.
# `path` stays on the read_file_sample handler for tests; the agent may not set it.
_APP_ARG_KEYS = frozenset(
    {
        "upload_dir",
        "cache_dir",
        "settings",
        "max_bytes",
        "max_full_load_rows",
        "include_postgres",
        "path",
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


def run_allowlisted_tool(
    name: str,
    arguments: dict[str, Any],
    settings: Settings,
    *,
    include_postgres: bool = False,
) -> dict[str, Any]:
    """Look up `name` on the registry, inject app args, and run the handler."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}.")
    llm_args = {key: value for key, value in arguments.items() if key not in _APP_ARG_KEYS}
    kwargs = _injected_kwargs(name, settings, llm_args, include_postgres)
    _LOG.info("execute tool=%s", name)
    return TOOL_REGISTRY[name].handler(**kwargs)


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
