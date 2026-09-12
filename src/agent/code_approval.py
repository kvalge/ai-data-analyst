# code_approval.py

"""HITL before generated SQL or Python runs. No UI imports; does not execute."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agent.json_output import strip_markdown_fences
from src.agent.state import HITL_MODE_GUIDED, HITL_MODE_STANDARD

KIND_APPROVE_CODE = "approve_code"

TOOL_RUN_ANALYSIS_CODE = "run_analysis_code"
TOOL_QUERY_DATABASE = "query_database"
CODE_TOOLS = frozenset({TOOL_RUN_ANALYSIS_CODE, TOOL_QUERY_DATABASE})

ACTION_APPROVE = "approve"
ACTION_EDIT_RUN = "edit_run"
ACTION_REJECT = "reject"

_PAUSE_MODES = frozenset({HITL_MODE_GUIDED, HITL_MODE_STANDARD})


def should_pause_generated_code(hitl_mode: str) -> bool:
    """Standard and Guided pause before generated SQL or Python. Auto does not."""
    return hitl_mode in _PAUSE_MODES


def generated_code_text(name: str, arguments: Mapping[str, Any]) -> str:
    """Return the SQL or Python text from a code-tool call. Empty if missing."""
    key = "sql" if name == TOOL_QUERY_DATABASE else "code"
    raw = arguments.get(key)
    if not isinstance(raw, str):
        return ""
    return raw.strip()


def source_id_for_code_tool(name: str, arguments: Mapping[str, Any]) -> str:
    """Identity shown in the interrupt. connection_id for SQL, else source_id."""
    key = "connection_id" if name == TOOL_QUERY_DATABASE else "source_id"
    raw = arguments.get(key)
    if not isinstance(raw, str):
        return ""
    return raw.strip()


def build_code_interrupt(
    *,
    tool: str,
    code: str,
    source_id: str,
    rationale: str = "",
) -> dict[str, Any]:
    """JSON-safe interrupt value. Code plus identity; never dataset rows."""
    return {
        "kind": KIND_APPROVE_CODE,
        "tool": tool,
        "code": code,
        "source_id": source_id,
        "rationale": rationale,
    }


def apply_code_decision(
    decision: object,
    *,
    name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Turn a resume value into execute args or an error. Reject does not run."""
    if not isinstance(decision, dict):
        return {
            "error": "Code approval decision is invalid.",
            "pending_tool": None,
            "pending_interrupt": None,
        }
    action = decision.get("action")
    if action == ACTION_REJECT:
        return {
            "error": "Generated code was rejected.",
            "pending_tool": None,
            "pending_interrupt": None,
        }
    if action == ACTION_APPROVE:
        return {
            "arguments": dict(arguments),
            "pending_interrupt": None,
            "error": None,
        }
    if action == ACTION_EDIT_RUN:
        edited = decision.get("code")
        if not isinstance(edited, str) or not edited.strip():
            return {
                "error": "Edited code is empty.",
                "pending_tool": None,
                "pending_interrupt": None,
            }
        updated = dict(arguments)
        field = "sql" if name == TOOL_QUERY_DATABASE else "code"
        updated[field] = strip_markdown_fences(edited)
        return {
            "arguments": updated,
            "pending_interrupt": None,
            "error": None,
        }
    return {
        "error": "Unknown code approval action.",
        "pending_tool": None,
        "pending_interrupt": None,
    }
