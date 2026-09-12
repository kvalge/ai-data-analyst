# __init__.py

"""Agent orchestration: parse helpers and the local Ollama client."""

from src.agent.json_output import (
    FAIL,
    RETRY_STRICT,
    STRICT_RETRY_INSTRUCTION,
    JsonOutputError,
    JsonParseError,
    JsonSchemaError,
    decide_after_parse_failure,
    parse_json_output,
    strip_markdown_fences,
)
from src.agent.llm import (
    ROLE_AGENTIC,
    ROLE_CODING,
    ROLE_FALLBACK_FAST,
    ROLE_PRIMARY,
    LlmError,
    complete,
    model_for_role,
)

__all__ = [
    "FAIL",
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
]
