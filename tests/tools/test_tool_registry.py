# test_tool_registry.py

"""Tests that every registered Phase 3 tool has a contract and handler."""

from collections.abc import Mapping
from typing import Any, cast

import pytest

from src.tools.list_sources import LIST_AVAILABLE_SOURCES, list_available_sources
from src.tools.load_full_file import LOAD_FULL_FILE, load_full_file
from src.tools.profile_source import PROFILE_SOURCE, profile_source
from src.tools.query_database import QUERY_DATABASE, query_database
from src.tools.read_sample import READ_FILE_SAMPLE, read_file_sample
from src.tools.registry import TOOL_REGISTRY, RegisteredTool, build_tool_registry

_EXPECTED_HANDLERS = {
    LIST_AVAILABLE_SOURCES.name: list_available_sources,
    READ_FILE_SAMPLE.name: read_file_sample,
    PROFILE_SOURCE.name: profile_source,
    LOAD_FULL_FILE.name: load_full_file,
    QUERY_DATABASE.name: query_database,
}


@pytest.fixture(scope="module")
def tool_registry() -> Mapping[str, RegisteredTool]:
    """Build the registry once; it is read-only so tests cannot interfere."""
    return build_tool_registry()


def test_phase_3_tool_names_are_registered(tool_registry: Mapping[str, RegisteredTool]):
    """The allowlist is list, sample, profile, load_full_file, and query_database."""
    assert set(tool_registry) == set(_EXPECTED_HANDLERS)


def test_registry_key_matches_contract_name(
    tool_registry: Mapping[str, RegisteredTool],
):
    """The dict key is the contract name, not a separate alias."""
    for name, entry in tool_registry.items():
        assert name == entry.contract.name


def test_every_registered_tool_has_description(
    tool_registry: Mapping[str, RegisteredTool],
):
    """The agent needs a non-empty description to choose a tool."""
    for entry in tool_registry.values():
        assert entry.contract.description.strip()


def test_every_registered_tool_has_input_schema(
    tool_registry: Mapping[str, RegisteredTool],
):
    """Each contract has a JSON-object input schema."""
    for entry in tool_registry.values():
        schema = entry.contract.input_schema
        assert schema["type"] == "object"
        assert "properties" in schema


def test_every_registered_tool_has_result_schema(
    tool_registry: Mapping[str, RegisteredTool],
):
    """Each contract has a JSON-object result schema."""
    for entry in tool_registry.values():
        schema = entry.contract.result_schema
        assert schema["type"] == "object"
        assert "properties" in schema


def test_registered_handlers_are_the_tool_functions(
    tool_registry: Mapping[str, RegisteredTool],
):
    """Handlers are the existing callables, not wrappers."""
    for name, handler in _EXPECTED_HANDLERS.items():
        assert tool_registry[name].handler is handler


def test_exported_registry_rejects_item_assignment():
    """TOOL_REGISTRY cannot be patched in place for the life of the process."""
    writable = cast(Any, TOOL_REGISTRY)
    with pytest.raises(TypeError):
        writable["profile_source"] = TOOL_REGISTRY["list_available_sources"]
