# test_load_approval.py

"""Tests for over-limit full-file load decisions. Reject must not load."""

from src.agent.load_approval import (
    ACTION_APPROVE,
    ACTION_REJECT,
    KIND_APPROVE_LOAD,
    apply_load_decision,
    build_load_interrupt,
    is_over_limit_load,
)
from src.tools.load_full_file import STATUS_LOADED, STATUS_NEEDS_APPROVAL


def _needs_approval() -> dict:
    return {
        "status": STATUS_NEEDS_APPROVAL,
        "source_id": "file-a",
        "reason": "row_count",
        "row_count": 5,
        "size_bytes": 80,
        "max_full_load_rows": 3,
        "max_bytes": 1000,
        "columns": [],
    }


def test_needs_approval_is_over_limit():
    """Only the needs_approval status trips the load interrupt."""
    assert is_over_limit_load(_needs_approval()) is True
    assert is_over_limit_load({"status": STATUS_LOADED}) is False


def test_payload_has_limits_not_rows():
    """The pause value is identity and limits, never dataset rows."""
    payload = build_load_interrupt(_needs_approval())
    assert payload["kind"] == KIND_APPROVE_LOAD
    assert payload["source_id"] == "file-a"
    assert payload["reason"] == "row_count"
    assert payload["row_count"] == 5
    assert payload["size_bytes"] == 80
    assert payload["max_full_load_rows"] == 3
    assert payload["max_bytes"] == 1000
    assert "columns" not in payload
    assert "rows" not in payload


def test_approve_clears_the_pause():
    """Approve is not an error and does not invent a source."""
    updates = apply_load_decision({"action": ACTION_APPROVE})
    assert updates["error"] is None


def test_reject_does_not_approve():
    """Reject is an error so execute_tool will not load."""
    updates = apply_load_decision({"action": ACTION_REJECT})
    assert updates["error"] == "Full-file load was rejected."


def test_unknown_action_is_rejected():
    """An unknown action is not treated as approve."""
    updates = apply_load_decision({"action": "ship_it"})
    assert updates["error"] == "Unknown full-file load action."
