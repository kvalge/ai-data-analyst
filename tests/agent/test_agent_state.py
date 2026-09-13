# test_agent_state.py

"""Tests that agent state is a TypedDict of summaries and paths, not frames."""

from pathlib import Path

import pytest

from src.agent.state import (
    ALLOWED_HITL_MODES,
    HITL_MODE_GUIDED,
    HITL_MODE_STANDARD,
    AgentState,
    as_artifact_path,
    empty_agent_state,
)


def test_empty_state_has_plan_keys():
    """The TypedDict keys are the Phase 3.4 state fields."""
    assert set(AgentState.__annotations__) == {
        "messages",
        "hitl_mode",
        "source_ids",
        "cleared_empty_source_ids",
        "cleared_profile_source_ids",
        "profile_summary",
        "pending_interrupt",
        "pending_tool",
        "last_tool_result",
        "artifacts",
        "error",
    }


def test_empty_state_defaults_to_standard():
    """A new thread starts in Standard unless a mode is passed."""
    assert empty_agent_state()["hitl_mode"] == HITL_MODE_STANDARD


def test_empty_state_accepts_guided_mode():
    """Guided is a valid starting mode."""
    assert empty_agent_state(hitl_mode=HITL_MODE_GUIDED)["hitl_mode"] == HITL_MODE_GUIDED


def test_unknown_hitl_mode_raises():
    """An unknown mode is not silently replaced with Standard."""
    with pytest.raises(ValueError, match="Unknown hitl_mode"):
        empty_agent_state(hitl_mode="turbo")


def test_empty_state_messages_start_empty():
    """A new thread has no conversation turns."""
    assert empty_agent_state()["messages"] == []


def test_empty_state_source_ids_start_empty():
    """Sources are attached later; the factory does not invent ids."""
    assert empty_agent_state()["source_ids"] == []


def test_empty_state_cleared_empty_ids_start_empty():
    """No source is treated as empty-schema-acked until the user confirms."""
    assert empty_agent_state()["cleared_empty_source_ids"] == []


def test_empty_state_cleared_profile_ids_start_empty():
    """No source is treated as already-profiled until schema/DQ/EDA finish."""
    assert empty_agent_state()["cleared_profile_source_ids"] == []


def test_empty_state_optional_payloads_are_none():
    """Profile, interrupt, tool result, and error start unset."""
    state = empty_agent_state()
    assert state["profile_summary"] is None
    assert state["pending_interrupt"] is None
    assert state["pending_tool"] is None
    assert state["last_tool_result"] is None
    assert state["error"] is None


def test_empty_state_artifacts_start_empty():
    """No artifact paths until a later node writes one."""
    assert empty_agent_state()["artifacts"] == []


def test_empty_state_lists_are_not_shared():
    """Mutating one new state must not change the next."""
    first = empty_agent_state()
    second = empty_agent_state()
    first["messages"].append({"role": "user", "content": "hi"})
    first["artifacts"].append("C:/tmp/chart.png")
    first["cleared_empty_source_ids"].append("file-abc")
    first["cleared_profile_source_ids"].append("file-abc")
    assert second["messages"] == []
    assert second["artifacts"] == []
    assert second["cleared_empty_source_ids"] == []
    assert second["cleared_profile_source_ids"] == []


def test_agent_state_annotations_omit_dataframe():
    """State types must not mention a dataframe."""
    rendered = " ".join(str(value) for value in AgentState.__annotations__.values())
    assert "DataFrame" not in rendered
    assert "ndarray" not in rendered


def test_as_artifact_path_stores_string(tmp_path: Path):
    """A Path is stored as a string path, not the file bytes."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    path = artifact_dir / "chart.png"
    assert as_artifact_path(path, artifact_dir=artifact_dir) == str(path.resolve())


def test_as_artifact_path_rejects_non_path(tmp_path: Path):
    """A non-path value is not coerced into an artifact."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    with pytest.raises(TypeError, match="filesystem paths"):
        as_artifact_path(123, artifact_dir=artifact_dir)  # type: ignore[arg-type]


def test_as_artifact_path_rejects_blank(tmp_path: Path):
    """An empty path string is not stored."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    with pytest.raises(ValueError, match="empty"):
        as_artifact_path("   ", artifact_dir=artifact_dir)


def test_as_artifact_path_rejects_outside_artifact_dir(tmp_path: Path):
    """A path that resolves outside ARTIFACT_DIR is not stored."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    sneaky = artifact_dir / ".." / "outside.png"
    # ValueError is the render-loop contract: a different base type would
    # miss ArtifactRenderError translation and crash the Streamlit rerun.
    with pytest.raises(ValueError, match="inside"):
        as_artifact_path(sneaky, artifact_dir=artifact_dir)


def test_allowed_hitl_modes_match_ui_labels():
    """Agent mode strings stay aligned with the Streamlit HITL labels."""
    assert ALLOWED_HITL_MODES == {"Guided", "Standard", "Auto"}
