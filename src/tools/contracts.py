# contracts.py

"""MCP-shaped in-process tool contracts (name, schemas, structured result).

Standing convention for every tool (1.13+ included):
- Success: return a plain dict that matches `result_schema`. No ok/error wrapper.
- Failure: raise a domain exception (`RegistryError`, `FileValidationError`, ...).
  The graph (Phase 3) maps that to a visible error; tools do not swallow it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolContract:
    """Description of one tool the agent may call. Does not run the tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    result_schema: dict[str, Any]


def empty_object_schema() -> dict[str, Any]:
    """JSON Schema for a tool that takes no caller arguments."""
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
