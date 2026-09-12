# test_code_approval.py

"""Tests for code-approval decisions. Reject must not imply a run."""

from src.agent.code_approval import (
    ACTION_APPROVE,
    ACTION_EDIT_RUN,
    ACTION_REJECT,
    KIND_APPROVE_CODE,
    TOOL_QUERY_DATABASE,
    TOOL_RUN_ANALYSIS_CODE,
    apply_code_decision,
    build_code_interrupt,
    generated_code_text,
    should_pause_generated_code,
    source_id_for_code_tool,
)
from src.agent.state import HITL_MODE_AUTO, HITL_MODE_GUIDED, HITL_MODE_STANDARD


def test_standard_and_guided_pause_before_code():
    """Auto skips the code interrupt; Standard and Guided do not."""
    assert should_pause_generated_code(HITL_MODE_STANDARD) is True
    assert should_pause_generated_code(HITL_MODE_GUIDED) is True
    assert should_pause_generated_code(HITL_MODE_AUTO) is False


def test_payload_has_code_source_and_rationale():
    """The pause value is code, source_id, and rationale — not dataset rows."""
    payload = build_code_interrupt(
        tool=TOOL_RUN_ANALYSIS_CODE,
        code="print(1)",
        source_id="file-a",
        rationale="sum revenue",
    )
    assert payload["kind"] == KIND_APPROVE_CODE
    assert payload["tool"] == TOOL_RUN_ANALYSIS_CODE
    assert payload["code"] == "print(1)"
    assert payload["source_id"] == "file-a"
    assert payload["rationale"] == "sum revenue"
    assert "rows" not in payload


def test_sql_tool_reads_sql_and_connection_id():
    """query_database shows sql as code and connection_id as source_id."""
    args = {"connection_id": "pg-1", "sql": "SELECT 1"}
    assert generated_code_text(TOOL_QUERY_DATABASE, args) == "SELECT 1"
    assert source_id_for_code_tool(TOOL_QUERY_DATABASE, args) == "pg-1"


def test_approve_keeps_original_arguments():
    """Approve returns the original code for execute."""
    args = {"source_id": "file-a", "code": "print(1)"}
    updates = apply_code_decision(
        {"action": ACTION_APPROVE},
        name=TOOL_RUN_ANALYSIS_CODE,
        arguments=args,
    )
    assert updates["error"] is None
    assert updates["arguments"] == args


def test_edit_run_replaces_code_and_strips_fences():
    """edit_run uses the submitted text, after fence strip."""
    updates = apply_code_decision(
        {"action": ACTION_EDIT_RUN, "code": "```python\nprint(7)\n```"},
        name=TOOL_RUN_ANALYSIS_CODE,
        arguments={"source_id": "file-a", "code": "print(1)"},
    )
    assert updates["error"] is None
    assert updates["arguments"]["code"] == "print(7)"


def test_edit_run_replaces_sql():
    """An SQL edit writes the sql argument, not a code key."""
    updates = apply_code_decision(
        {"action": ACTION_EDIT_RUN, "code": "SELECT revenue FROM sales"},
        name=TOOL_QUERY_DATABASE,
        arguments={"connection_id": "pg-1", "sql": "SELECT 1"},
    )
    assert updates["arguments"]["sql"] == "SELECT revenue FROM sales"
    assert "code" not in updates["arguments"]


def test_reject_does_not_return_arguments():
    """Reject is an error and does not hand arguments to execute."""
    updates = apply_code_decision(
        {"action": ACTION_REJECT},
        name=TOOL_RUN_ANALYSIS_CODE,
        arguments={"source_id": "file-a", "code": "print(1)"},
    )
    assert updates["error"] == "Generated code was rejected."
    assert "arguments" not in updates


def test_empty_edit_is_rejected():
    """Blank edited code is not executed."""
    updates = apply_code_decision(
        {"action": ACTION_EDIT_RUN, "code": "   "},
        name=TOOL_RUN_ANALYSIS_CODE,
        arguments={"source_id": "file-a", "code": "print(1)"},
    )
    assert updates["error"] == "Edited code is empty."
    assert "arguments" not in updates


def test_unknown_action_is_rejected():
    """An unknown action is not treated as approve."""
    updates = apply_code_decision(
        {"action": "ship_it"},
        name=TOOL_RUN_ANALYSIS_CODE,
        arguments={"source_id": "file-a", "code": "print(1)"},
    )
    assert updates["error"] == "Unknown code approval action."
    assert "arguments" not in updates
