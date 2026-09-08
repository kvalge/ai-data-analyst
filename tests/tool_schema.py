# tool_schema.py

"""Shared assertions for MCP-shaped tool result schemas."""

from __future__ import annotations

from collections.abc import Mapping


def assert_keys_match_required(
    payload: Mapping[str, object], schema: Mapping[str, object]
) -> None:
    """Require payload keys to equal the schema's `required` list."""
    required = schema["required"]
    assert isinstance(required, list)
    assert set(payload) == set(required)
