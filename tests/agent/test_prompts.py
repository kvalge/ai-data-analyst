# test_prompts.py

"""Tests that the system prompt covers safety, tools, and HITL without UI imports."""

from pathlib import Path

import pytest

from src.agent.prompts import build_system_prompt
from src.agent.state import HITL_MODE_AUTO, HITL_MODE_GUIDED, HITL_MODE_STANDARD
from src.tools.registry import TOOL_REGISTRY

_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "src" / "agent" / "prompts.py"


def test_prompts_module_does_not_import_streamlit():
    """System prompts live in the agent layer, not the UI."""
    source = _PROMPTS_PATH.read_text(encoding="utf-8")
    assert "streamlit" not in source.lower()


def test_prompt_keeps_inference_local():
    """The model is told not to send data off this machine."""
    text = build_system_prompt(hitl_mode=HITL_MODE_STANDARD)
    assert "this machine" in text
    assert "cloud" in text.lower()


def test_prompt_forbids_dumping_full_data():
    """Full datasets must not be pasted into replies or tool arguments."""
    text = build_system_prompt(hitl_mode=HITL_MODE_STANDARD)
    assert "Never dump a full dataset" in text


def test_prompt_tells_model_to_reuse_artifact_paths():
    """Follow-ups should reuse listed artifact files, not invent paths."""
    text = build_system_prompt(hitl_mode=HITL_MODE_STANDARD)
    assert "reuse those files for follow-up" in text
    assert "Do not invent artifact paths" in text


def test_prompt_forbids_inventing_tools():
    """The model must not invent a tool name."""
    text = build_system_prompt(hitl_mode=HITL_MODE_STANDARD)
    assert "Do not invent a tool name" in text


def test_prompt_describes_tool_json_shape():
    """A tool call is a JSON object; a user answer is plain text."""
    text = build_system_prompt(hitl_mode=HITL_MODE_STANDARD)
    assert '{"name": "<tool>", "arguments": {}}' in text
    assert "plain text" in text


def test_prompt_lists_registered_tools():
    """Every Phase 3 registry name appears so the allowlist stays in sync."""
    text = build_system_prompt(hitl_mode=HITL_MODE_STANDARD)
    for name in TOOL_REGISTRY:
        assert name in text


def test_prompt_does_not_hardcode_a_model_name():
    """Model tags stay in env settings, not in the prompt text."""
    text = build_system_prompt(hitl_mode=HITL_MODE_STANDARD).lower()
    assert "qwen" not in text
    assert "llama" not in text


def test_unknown_hitl_mode_raises():
    """An unknown mode is not silently replaced with Standard."""
    with pytest.raises(ValueError, match="Unknown hitl_mode"):
        build_system_prompt(hitl_mode="turbo")


def test_guided_pauses_after_profiling_steps():
    """Guided waits after schema, DQ, and EDA."""
    text = build_system_prompt(hitl_mode=HITL_MODE_GUIDED)
    assert "After schema, after data-quality, and after EDA" in text


def test_standard_pauses_before_generated_code():
    """Standard still waits before SQL or Python runs."""
    text = build_system_prompt(hitl_mode=HITL_MODE_STANDARD)
    assert "Before running generated SQL or Python" in text
    assert "Profiling may run through without pausing" in text


def test_auto_may_run_code_without_asking():
    """Auto may run generated code without a separate load-line (shared covers it)."""
    text = build_system_prompt(hitl_mode=HITL_MODE_AUTO)
    assert "may run without asking" in text


def test_all_modes_wait_on_oversize_full_load():
    """Over-limit full-file load is a pause in every mode."""
    shared = "Before a full-file load over the configured size or row limit"
    for mode in (HITL_MODE_GUIDED, HITL_MODE_STANDARD, HITL_MODE_AUTO):
        assert shared in build_system_prompt(hitl_mode=mode)


def test_default_mode_is_standard():
    """Omitting hitl_mode uses Standard. This is the only default-omission test."""
    assert build_system_prompt() == build_system_prompt(hitl_mode=HITL_MODE_STANDARD)
