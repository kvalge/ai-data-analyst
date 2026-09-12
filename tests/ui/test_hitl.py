# test_hitl.py

"""Tests for HITL session-mode defaults and persistence."""

from src.ui.hitl import (
    DEFAULT_HITL_MODE,
    HITL_MODE_KEY,
    HitlMode,
    ensure_hitl_mode,
    resolve_hitl_mode,
)


def test_resolve_defaults_to_standard():
    """Missing or invalid values become Standard."""
    assert resolve_hitl_mode(None) == DEFAULT_HITL_MODE
    assert resolve_hitl_mode("turbo") == HitlMode.STANDARD


def test_resolve_keeps_valid_modes():
    """Guided, Standard, and Auto are accepted as-is."""
    assert resolve_hitl_mode("Guided") == HitlMode.GUIDED
    assert resolve_hitl_mode(HitlMode.STANDARD) == HitlMode.STANDARD
    assert resolve_hitl_mode("Auto") == HitlMode.AUTO


def test_ensure_persists_across_reruns():
    """A chosen mode stays in session_state on the next ensure call."""
    session_state: dict[str, object] = {}
    assert ensure_hitl_mode(session_state) == HitlMode.STANDARD

    session_state[HITL_MODE_KEY] = HitlMode.GUIDED
    assert ensure_hitl_mode(session_state) == HitlMode.GUIDED
    assert session_state[HITL_MODE_KEY] == HitlMode.GUIDED


def test_ensure_overwrites_invalid_mode():
    """A leftover bad value is replaced with the default in session_state."""
    session_state: dict[str, object] = {HITL_MODE_KEY: "not-a-real-mode"}

    assert ensure_hitl_mode(session_state) == DEFAULT_HITL_MODE
    assert session_state[HITL_MODE_KEY] == DEFAULT_HITL_MODE
