# test_code_review.py

"""Tests for the code-review decision helper. The widget does not run code."""

import inspect
from pathlib import Path

import src.ui.code_review as code_review
from src.agent.code_approval import ACTION_APPROVE, ACTION_EDIT_RUN, ACTION_REJECT
from src.ui.code_review import (
    code_review_decision,
    code_review_widget_keys,
    forget_code_review_session,
    sync_code_review_session,
)

_CODE_REVIEW_PATH = Path(inspect.getfile(code_review)).resolve()


def test_unchanged_approve_keeps_original():
    """Approve with the same text does not send an edit."""
    assert code_review_decision(
        action=ACTION_APPROVE,
        original="print(1)",
        edited="print(1)",
    ) == {"action": ACTION_APPROVE}


def test_changed_approve_is_edit_run():
    """Approve after an edit sends the textarea, not the original snippet."""
    assert code_review_decision(
        action=ACTION_APPROVE,
        original="print(1)",
        edited="print(7)",
    ) == {"action": ACTION_EDIT_RUN, "code": "print(7)"}


def test_reject_does_not_include_code():
    """Reject ignores the textarea and does not look like a run."""
    assert code_review_decision(
        action=ACTION_REJECT,
        original="print(1)",
        edited="print(7)",
    ) == {"action": ACTION_REJECT}


def test_widget_keys_change_with_source_or_code():
    """Two pauses that share a filename-looking id still get distinct widget keys."""
    first = {
        "tool": "run_analysis_code",
        "source_id": "file-a",
        "code": "print(1)",
    }
    other_source = {**first, "source_id": "file-b"}
    other_code = {**first, "code": "print(2)"}
    assert code_review_widget_keys(first) != code_review_widget_keys(other_source)
    assert code_review_widget_keys(first) != code_review_widget_keys(other_code)
    assert code_review_widget_keys(first) == code_review_widget_keys(first)


def test_same_payload_keeps_in_progress_edit():
    """A rerun of the same interrupt does not wipe the textarea."""
    payload = {
        "tool": "run_analysis_code",
        "source_id": "file-a",
        "code": "print(1)",
    }
    session: dict[str, object] = {}
    keys = sync_code_review_session(session, payload)
    session[keys["text"]] = "print(7)"
    again = sync_code_review_session(session, payload)
    assert again == keys
    assert session[keys["text"]] == "print(7)"


def test_new_payload_drops_stale_textarea():
    """A later interrupt does not keep the previous pause's edited text."""
    first = {
        "tool": "run_analysis_code",
        "source_id": "file-a",
        "code": "print(1)",
    }
    second = {
        "tool": "run_analysis_code",
        "source_id": "file-b",
        "code": "print(2)",
    }
    session: dict[str, object] = {}
    keys = sync_code_review_session(session, first)
    session[keys["text"]] = "print(7)"
    next_keys = sync_code_review_session(session, second)
    assert next_keys != keys
    assert keys["text"] not in session
    assert next_keys["text"] not in session


def test_forget_after_decision_clears_same_payload():
    """A later pause with the same generated code is not a leftover edit."""
    payload = {
        "tool": "run_analysis_code",
        "source_id": "file-a",
        "code": "print(1)",
    }
    session: dict[str, object] = {}
    keys = sync_code_review_session(session, payload)
    session[keys["text"]] = "print(7)"
    forget_code_review_session(session, keys)
    later = sync_code_review_session(session, payload)
    assert later["text"] not in session


def test_code_review_module_does_not_run_code():
    """The widget module must not execute or import the code tools."""
    source = _CODE_REVIEW_PATH.read_text(encoding="utf-8")
    assert "run_analysis_code" not in source
    assert "query_database" not in source
    assert "exec(" not in source
    assert "eval(" not in source
    assert "subprocess" not in source
