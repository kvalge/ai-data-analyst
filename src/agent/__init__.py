# __init__.py

"""Agent orchestration: parse helpers, Ollama client, state, prompts, and graph."""

from src.agent.json_output import (
    FAIL,
    FIRST_PARSE_FAILURE,
    RETRY_STRICT,
    STRICT_RETRY_INSTRUCTION,
    JsonOutputError,
    JsonParseError,
    JsonSchemaError,
    decide_after_parse_failure,
    parse_json_output,
    strip_markdown_fences,
)
from src.agent.confirm_sources import (
    ACTION_ABORT,
    ACTION_CONFIRM,
    ACTION_SELECT,
    KIND_CONFIRM_SOURCES,
    apply_confirm_decision,
    build_interrupt_payload,
    decide_confirm_reason,
)
from src.agent.execute import (
    ToolValidationError,
    interpret_model_reply,
    parse_tool_call,
    retry_prompt_after_validation,
    run_allowlisted_tool,
    validate_tool_call,
)
from src.agent.graph import CompleteFn, build_graph, build_prompt
from src.agent.llm import (
    ROLE_AGENTIC,
    ROLE_CODING,
    ROLE_FALLBACK_FAST,
    ROLE_PRIMARY,
    LlmError,
    complete,
    model_for_role,
)
from src.agent.prompts import build_system_prompt
from src.agent.state import (
    ALLOWED_HITL_MODES,
    HITL_MODE_AUTO,
    HITL_MODE_GUIDED,
    HITL_MODE_STANDARD,
    AgentMessage,
    AgentState,
    as_artifact_path,
    empty_agent_state,
)

__all__ = [
    "FAIL",
    "FIRST_PARSE_FAILURE",
    "RETRY_STRICT",
    "STRICT_RETRY_INSTRUCTION",
    "JsonOutputError",
    "JsonParseError",
    "JsonSchemaError",
    "decide_after_parse_failure",
    "parse_json_output",
    "strip_markdown_fences",
    "ROLE_AGENTIC",
    "ROLE_CODING",
    "ROLE_FALLBACK_FAST",
    "ROLE_PRIMARY",
    "LlmError",
    "complete",
    "model_for_role",
    "ALLOWED_HITL_MODES",
    "HITL_MODE_AUTO",
    "HITL_MODE_GUIDED",
    "HITL_MODE_STANDARD",
    "AgentMessage",
    "AgentState",
    "as_artifact_path",
    "empty_agent_state",
    "build_system_prompt",
    "CompleteFn",
    "ACTION_ABORT",
    "ACTION_CONFIRM",
    "ACTION_SELECT",
    "KIND_CONFIRM_SOURCES",
    "apply_confirm_decision",
    "build_interrupt_payload",
    "decide_confirm_reason",
    "build_graph",
    "build_prompt",
    "ToolValidationError",
    "interpret_model_reply",
    "parse_tool_call",
    "retry_prompt_after_validation",
    "run_allowlisted_tool",
    "validate_tool_call",
]
