# test_profile_step.py

"""Tests for Guided profile section pauses (UI stepper, not LangGraph)."""

from src.ui.hitl import HitlMode
from src.ui.profile_step import (
    PROFILE_SECTIONS,
    SECTION_DQ,
    SECTION_EDA,
    SECTION_SCHEMA,
    continue_profile_step,
    initial_profile_step,
    profile_stepper_active,
    resolve_profile_step,
    skip_remaining_profile_step,
    visible_profile_sections,
)


def test_guided_starts_at_schema():
    """Guided reveals schema first and waits."""
    assert initial_profile_step(HitlMode.GUIDED) == SECTION_SCHEMA


def test_standard_starts_at_eda():
    """Standard shows every section with no pause."""
    assert initial_profile_step(HitlMode.STANDARD) == SECTION_EDA


def test_auto_starts_at_eda():
    """Auto shows every section with no pause."""
    assert initial_profile_step(HitlMode.AUTO) == SECTION_EDA


def test_continue_schema_reveals_dq():
    """Continue after schema uncovers data quality."""
    assert continue_profile_step(SECTION_SCHEMA) == SECTION_DQ


def test_continue_dq_reveals_eda():
    """Continue after DQ uncovers EDA."""
    assert continue_profile_step(SECTION_DQ) == SECTION_EDA


def test_continue_at_eda_stays_eda():
    """There is no section after EDA."""
    assert continue_profile_step(SECTION_EDA) == SECTION_EDA


def test_continue_follows_profile_sections_order():
    """Continue is the next PROFILE_SECTIONS entry, not a hand-written chain."""
    for index, section in enumerate(PROFILE_SECTIONS[:-1]):
        assert continue_profile_step(section) == PROFILE_SECTIONS[index + 1]
    assert continue_profile_step(PROFILE_SECTIONS[-1]) == PROFILE_SECTIONS[-1]


def test_skip_remaining_jumps_to_eda():
    """Skip remaining reveals every later section at once."""
    assert skip_remaining_profile_step() == SECTION_EDA


def test_visible_sections_schema_only():
    """The first pause shows schema and hides DQ/EDA."""
    assert visible_profile_sections(SECTION_SCHEMA) == (SECTION_SCHEMA,)


def test_visible_sections_through_dq():
    """After one Continue, schema stays and DQ appears."""
    assert visible_profile_sections(SECTION_DQ) == (SECTION_SCHEMA, SECTION_DQ)


def test_visible_sections_all():
    """The last step shows schema, DQ, and EDA."""
    assert visible_profile_sections(SECTION_EDA) == PROFILE_SECTIONS


def test_visible_sections_are_a_prefix_of_profile_sections():
    """Visible sections are PROFILE_SECTIONS[:index+1], not a per-step branch."""
    for index, section in enumerate(PROFILE_SECTIONS):
        assert visible_profile_sections(section) == PROFILE_SECTIONS[: index + 1]


def test_resolve_invalid_step_is_schema():
    """A bad stored step pauses at schema instead of skipping ahead."""
    assert resolve_profile_step("nope") == SECTION_SCHEMA


def test_resolve_missing_step_is_schema():
    """A missing step pauses at schema."""
    assert resolve_profile_step(None) == SECTION_SCHEMA


def test_stepper_active_guided_schema():
    """Guided at schema still has later sections."""
    assert profile_stepper_active(HitlMode.GUIDED, SECTION_SCHEMA) is True


def test_stepper_active_guided_dq():
    """Guided at DQ still has EDA left."""
    assert profile_stepper_active(HitlMode.GUIDED, SECTION_DQ) is True


def test_stepper_active_guided_eda():
    """Guided at EDA has nothing left to reveal."""
    assert profile_stepper_active(HitlMode.GUIDED, SECTION_EDA) is False


def test_stepper_inactive_in_standard():
    """Standard never shows the stepper; mode decides, not the stored step."""
    for section in PROFILE_SECTIONS:
        assert profile_stepper_active(HitlMode.STANDARD, section) is False


def test_stepper_inactive_in_auto():
    """Auto never shows the stepper; mode decides, not the stored step."""
    for section in PROFILE_SECTIONS:
        assert profile_stepper_active(HitlMode.AUTO, section) is False
