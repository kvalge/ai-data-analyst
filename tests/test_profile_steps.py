# test_profile_steps.py

"""Tests for Guided profile-step decisions. No dataset rows in payloads."""

from src.agent.profile_steps import (
    ACTION_ABORT,
    ACTION_CONTINUE,
    ACTION_SKIP_REMAINING,
    KIND_PROFILE_STEP,
    PROFILE_SECTIONS,
    SECTION_DQ,
    SECTION_EDA,
    SECTION_SCHEMA,
    apply_profile_decision,
    build_profile_interrupt,
    should_pause_profiling,
    visible_profile_sections,
    with_cleared_profile_ids,
)
from src.agent.state import HITL_MODE_AUTO, HITL_MODE_GUIDED, HITL_MODE_STANDARD


def _summary() -> dict:
    return {
        "source_id": "file-a",
        "cached": True,
        "sample_row_count": 3,
        "schema": {"columns": ["date", "region", "revenue"], "dtypes": {}},
        "dq": {"duplicates": {"duplicate_row_count": 0}},
        "eda": {"summary": {}},
    }


def test_only_guided_pauses_profiling():
    """Standard and Auto run schema / DQ / EDA without a pause."""
    assert should_pause_profiling(HITL_MODE_GUIDED) is True
    assert should_pause_profiling(HITL_MODE_STANDARD) is False
    assert should_pause_profiling(HITL_MODE_AUTO) is False


def test_interrupt_payload_is_section_only():
    """The pause value is identities plus the compact section, not rows."""
    payload = build_profile_interrupt(SECTION_SCHEMA, _summary())
    assert payload["kind"] == KIND_PROFILE_STEP
    assert payload["step"] == SECTION_SCHEMA
    assert payload["source_id"] == "file-a"
    assert payload["section"]["columns"] == ["date", "region", "revenue"]
    assert "rows" not in payload
    assert "stored_path" not in payload
    dumped = str(payload)
    assert "2024-01-01" not in dumped


def test_continue_does_not_clear_source():
    """Continue leaves later Guided pauses in place."""
    updates = apply_profile_decision(
        {"action": ACTION_CONTINUE},
        source_id="file-a",
        cleared_profile_source_ids=[],
    )
    assert updates["error"] is None
    assert "cleared_profile_source_ids" not in updates


def test_skip_remaining_records_source_once():
    """Skip remaining marks the source so later nodes do not pause again."""
    updates = apply_profile_decision(
        {"action": ACTION_SKIP_REMAINING},
        source_id="file-a",
        cleared_profile_source_ids=["file-a"],
    )
    assert updates["cleared_profile_source_ids"] == ["file-a"]


def test_abort_does_not_clear_source():
    """Abort ends the turn and does not treat the source as profiled."""
    updates = apply_profile_decision(
        {"action": ACTION_ABORT},
        source_id="file-a",
        cleared_profile_source_ids=[],
    )
    assert updates["error"] == "Profiling was aborted."
    assert "cleared_profile_source_ids" not in updates


def test_unknown_action_is_visible():
    """An unknown resume action is not treated as continue."""
    updates = apply_profile_decision(
        {"action": "approve"},
        source_id="file-a",
        cleared_profile_source_ids=[],
    )
    assert updates["error"] == "Unknown profiling action."


def test_invalid_decision_is_visible():
    """A non-dict resume value is not guessed into continue."""
    updates = apply_profile_decision(
        "continue",
        source_id="file-a",
        cleared_profile_source_ids=[],
    )
    assert updates["error"] == "Profiling decision is invalid."


def test_cleared_ids_copy_caller_list():
    """Appending a cleared id must not mutate the caller's list."""
    existing = ["file-old"]
    merged = with_cleared_profile_ids(existing, ["file-a"])
    assert merged == ["file-old", "file-a"]
    assert existing == ["file-old"]


def test_visible_sections_keep_earlier_ones():
    """A DQ pause still shows schema."""
    assert visible_profile_sections(SECTION_DQ) == (
        SECTION_SCHEMA,
        SECTION_DQ,
    )


def test_visible_sections_schema_only():
    """The first pause shows schema and hides DQ/EDA."""
    assert visible_profile_sections(SECTION_SCHEMA) == (SECTION_SCHEMA,)


def test_visible_sections_all():
    """The last step shows schema, DQ, and EDA."""
    assert visible_profile_sections(SECTION_EDA) == PROFILE_SECTIONS


def test_visible_sections_are_a_prefix_of_profile_sections():
    """Visible sections are PROFILE_SECTIONS[:index+1], not a per-step branch."""
    for index, section in enumerate(PROFILE_SECTIONS):
        assert visible_profile_sections(section) == PROFILE_SECTIONS[: index + 1]


def test_visible_sections_unknown_step_is_schema():
    """A bad step name pauses at schema instead of skipping ahead."""
    assert visible_profile_sections("nope") == (SECTION_SCHEMA,)
