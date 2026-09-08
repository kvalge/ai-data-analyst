# hitl.py

"""HITL session mode. Stored in Streamlit session_state until the graph exists."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class HitlMode(StrEnum):
    """Session HITL mode. Values are the UI labels."""

    GUIDED = "Guided"
    STANDARD = "Standard"
    AUTO = "Auto"


DEFAULT_HITL_MODE = HitlMode.STANDARD
HITL_MODE_KEY = "hitl_mode"


def resolve_hitl_mode(stored: object | None) -> HitlMode:
    """Return a valid mode, defaulting to Standard."""
    if isinstance(stored, HitlMode):
        return stored
    try:
        return HitlMode(stored)
    except (TypeError, ValueError):
        return DEFAULT_HITL_MODE


def ensure_hitl_mode(session_state: Any) -> HitlMode:
    """Write a valid HITL mode into session_state and return it.

    `session_state` is Streamlit's proxy or a plain dict in tests.
    """
    mode = resolve_hitl_mode(session_state.get(HITL_MODE_KEY))
    session_state[HITL_MODE_KEY] = mode
    return mode
