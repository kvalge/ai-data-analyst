# load_approval.py

"""HITL before an over-limit full-file load. No UI imports; does not load."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.tools.load_full_file import STATUS_NEEDS_APPROVAL

KIND_APPROVE_LOAD = "approve_load"

ACTION_APPROVE = "approve"
ACTION_REJECT = "reject"


def is_over_limit_load(result: Mapping[str, Any]) -> bool:
    """True when load_full_file asked for approval instead of loading."""
    return result.get("status") == STATUS_NEEDS_APPROVAL


def build_load_interrupt(result: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe interrupt value. Limits and identity; never dataset rows."""
    row_count = result.get("row_count")
    return {
        "kind": KIND_APPROVE_LOAD,
        "source_id": str(result.get("source_id") or ""),
        "reason": str(result.get("reason") or ""),
        "row_count": row_count if isinstance(row_count, int) else None,
        "size_bytes": int(result.get("size_bytes") or 0),
        "max_full_load_rows": int(result.get("max_full_load_rows") or 0),
        "max_bytes": int(result.get("max_bytes") or 0),
    }


def apply_load_decision(decision: object) -> dict[str, Any]:
    """Turn a resume value into approve or an error. Reject does not load."""
    if not isinstance(decision, dict):
        return {
            "error": "Full-file load decision is invalid.",
            "pending_tool": None,
            "pending_interrupt": None,
        }
    action = decision.get("action")
    if action == ACTION_REJECT:
        return {
            "error": "Full-file load was rejected.",
            "pending_tool": None,
            "pending_interrupt": None,
        }
    if action == ACTION_APPROVE:
        return {"pending_interrupt": None, "error": None}
    return {
        "error": "Unknown full-file load action.",
        "pending_tool": None,
        "pending_interrupt": None,
    }
