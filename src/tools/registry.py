# registry.py

"""Register in-process tools: name → contract + handler.

The agent (3.8) looks up tools here. This module does not run them or
inject app arguments. Tools: list, sample, profile, load_full_file,
query_database, run_analysis_code, retrieve_domain_context.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from src.tools.contracts import ToolContract
from src.tools.list_sources import LIST_AVAILABLE_SOURCES, list_available_sources
from src.tools.load_full_file import LOAD_FULL_FILE, load_full_file
from src.tools.profile_source import PROFILE_SOURCE, profile_source
from src.tools.query_database import QUERY_DATABASE, query_database
from src.tools.read_sample import READ_FILE_SAMPLE, read_file_sample
from src.tools.retrieve_domain_context import (
    RETRIEVE_DOMAIN_CONTEXT,
    retrieve_domain_context,
)
from src.tools.run_analysis_code import RUN_ANALYSIS_CODE, run_analysis_code

ToolHandler = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class RegisteredTool:
    """A contract plus the function that implements it."""

    contract: ToolContract
    handler: ToolHandler


_TOOL_ENTRIES: tuple[tuple[ToolContract, ToolHandler], ...] = (
    (LIST_AVAILABLE_SOURCES, list_available_sources),
    (READ_FILE_SAMPLE, read_file_sample),
    (PROFILE_SOURCE, profile_source),
    (LOAD_FULL_FILE, load_full_file),
    (QUERY_DATABASE, query_database),
    (RUN_ANALYSIS_CODE, run_analysis_code),
    (RETRIEVE_DOMAIN_CONTEXT, retrieve_domain_context),
)


def build_tool_registry() -> Mapping[str, RegisteredTool]:
    """Return a read-only name → contract + handler map. Duplicate names raise."""
    registry: dict[str, RegisteredTool] = {}
    for contract, handler in _TOOL_ENTRIES:
        if contract.name in registry:
            raise ValueError(f"Duplicate tool name: {contract.name}")
        registry[contract.name] = RegisteredTool(contract=contract, handler=handler)
    return MappingProxyType(registry)


TOOL_REGISTRY = build_tool_registry()
