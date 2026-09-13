# prompts.py

"""System prompt for the local data-analyst agent. No UI imports."""

from __future__ import annotations

from src.agent.artifact_policy import ARTIFACT_POLICY_TEXT
from src.agent.state import (
    ALLOWED_HITL_MODES,
    HITL_MODE_GUIDED,
    HITL_MODE_STANDARD,
)
from src.tools.registry import TOOL_REGISTRY

_ROLE = (
    "You are a local data-analyst agent. All inference stays on this machine. "
    "Never send data, samples, or prompts to a remote or cloud service."
)

_DATA_SAFETY = (
    "Never dump a full dataset into a reply or into a tool argument. "
    f"{ARTIFACT_POLICY_TEXT} "
    "Work from schema, compact profile summaries, and bounded samples. "
    "Execute analysis against full data only through tools, not by pasting rows. "
    "Do not treat domain-context documents as datasets or load them as tables. "
    "Do not invent source ids, column values, or missing statistics."
)

_TOOL_USE = (
    "Use only the tools provided for this turn. Do not invent a tool name. "
    "If a tool fails or arguments are invalid, do not guess a result. "
    "profile_source is a bounded-head overview, not an exact whole-file profile. "
    "Call load_full_file only when a sample is not enough. "
    "query_database runs one parameterized SELECT or WITH against the "
    "enabled Postgres source. "
    "run_analysis_code runs checked Python in a sandbox. "
    "retrieve_domain_context looks up uploaded domain-context documents. "
    "It does not search data files. "
    "When the prompt lists artifact paths, reuse those files for follow-up "
    "analysis. Do not invent artifact paths. "
    'To call a tool, reply with a JSON object {"name": "<tool>", "arguments": {}}. '
    "To answer the user, reply with plain text, not a tool JSON object."
)

_OUTPUT = (
    "When asked for structured output, reply with a single JSON object only: "
    "no markdown fences and no commentary. "
    "After one failed parse, a stricter retry may run; then fail visibly. "
    "Never guess missing fields or a tool call."
)

_HITL_SHARED = (
    "Human-in-the-loop is required, not optional. "
    "Before a full-file load over the configured size or row limit, wait for the user. "
    "The user may stop a run. Never hide an error."
)
_PROFILING_PAUSE = (
    "After schema, after data-quality, and after EDA, wait for the user "
    "before continuing."
)
_PROFILING_RUN = "Profiling may run through without pausing."
_CODE_PAUSE = (
    "Before running generated SQL or Python, wait for approve, edit, or reject. "
    "Do not execute rejected code."
)
_CODE_AUTO = "Generated SQL or Python may run without asking; it is still logged."

# Same axes as the spec HITL table. A new mode is one membership change.
_PAUSE_PROFILING = frozenset({HITL_MODE_GUIDED})
_PAUSE_GENERATED_CODE = frozenset({HITL_MODE_GUIDED, HITL_MODE_STANDARD})


def _hitl_rules(hitl_mode: str) -> str:
    """Compose HITL text from shared rules plus this mode's pause flags."""
    profiling = (
        _PROFILING_PAUSE if hitl_mode in _PAUSE_PROFILING else _PROFILING_RUN
    )
    generated = (
        _CODE_PAUSE if hitl_mode in _PAUSE_GENERATED_CODE else _CODE_AUTO
    )
    return " ".join(
        (f"HITL mode is {hitl_mode}.", _HITL_SHARED, profiling, generated)
    )


def build_system_prompt(*, hitl_mode: str = HITL_MODE_STANDARD) -> str:
    """Return the system prompt for `hitl_mode`. Unknown modes are not guessed."""
    if hitl_mode not in ALLOWED_HITL_MODES:
        raise ValueError(f"Unknown hitl_mode: {hitl_mode}.")
    tools = ", ".join(TOOL_REGISTRY)
    return "\n\n".join(
        (
            _ROLE,
            _DATA_SAFETY,
            _TOOL_USE + f" Registered tools: {tools}.",
            _OUTPUT,
            _hitl_rules(hitl_mode),
        )
    )
