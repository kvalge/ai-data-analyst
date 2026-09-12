# profile_step.py

"""UI-only Guided pauses for profile sections. Not a LangGraph interrupt.

profile_source still computes the compact summary in one pass. This module
only decides which sections to show and how Continue / Skip remaining /
Abort move that reveal. Graph nodes come in 3.11.
"""

from __future__ import annotations

from src.ui.hitl import HitlMode

SECTION_SCHEMA = "schema"
SECTION_DQ = "dq"
SECTION_EDA = "eda"
# Append here to add a Guided pause. Continue / skip / visible slices follow this tuple.
PROFILE_SECTIONS = (SECTION_SCHEMA, SECTION_DQ, SECTION_EDA)

PROFILE_STEP_KEY = "profile_step"
ACTION_CONTINUE = "continue"
ACTION_SKIP_REMAINING = "skip_remaining"
ACTION_ABORT = "abort"


def initial_profile_step(mode: HitlMode) -> str:
    """Guided starts at the first section; Standard and Auto show every section."""
    if mode == HitlMode.GUIDED:
        return PROFILE_SECTIONS[0]
    return PROFILE_SECTIONS[-1]


def resolve_profile_step(stored: object | None) -> str:
    """Return a valid section step, defaulting to the first (more pause)."""
    if stored in PROFILE_SECTIONS:
        return str(stored)
    return PROFILE_SECTIONS[0]


def continue_profile_step(step: str) -> str:
    """Advance one section. Already at the last section stays there."""
    current = resolve_profile_step(step)
    index = PROFILE_SECTIONS.index(current)
    return PROFILE_SECTIONS[min(index + 1, len(PROFILE_SECTIONS) - 1)]


def skip_remaining_profile_step() -> str:
    """Show every remaining section."""
    return PROFILE_SECTIONS[-1]


def visible_profile_sections(step: str) -> tuple[str, ...]:
    """Sections to render through this pause point (earlier ones stay visible)."""
    current = resolve_profile_step(step)
    return PROFILE_SECTIONS[: PROFILE_SECTIONS.index(current) + 1]


def profile_stepper_active(mode: HitlMode, step: str) -> bool:
    """True when Guided still has a later section to reveal."""
    return (
        mode == HitlMode.GUIDED
        and resolve_profile_step(step) != PROFILE_SECTIONS[-1]
    )
