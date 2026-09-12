# __init__.py

"""In-process data-access tools with MCP-shaped contracts."""

from src.tools.contracts import ToolContract
from src.tools.list_sources import LIST_AVAILABLE_SOURCES, list_available_sources
from src.tools.load_full_file import LOAD_FULL_FILE, load_full_file
from src.tools.profile_source import PROFILE_SOURCE, ProfileError, profile_source
from src.tools.query_database import QUERY_DATABASE, query_database
from src.tools.read_sample import READ_FILE_SAMPLE, read_file_sample
from src.tools.run_analysis_code import (
    RUN_ANALYSIS_CODE,
    AnalysisCodeError,
    run_analysis_code,
)
from src.tools.registry import TOOL_REGISTRY, RegisteredTool, build_tool_registry

__all__ = [
    "LIST_AVAILABLE_SOURCES",
    "LOAD_FULL_FILE",
    "PROFILE_SOURCE",
    "ProfileError",
    "QUERY_DATABASE",
    "READ_FILE_SAMPLE",
    "RUN_ANALYSIS_CODE",
    "AnalysisCodeError",
    "RegisteredTool",
    "TOOL_REGISTRY",
    "ToolContract",
    "build_tool_registry",
    "list_available_sources",
    "load_full_file",
    "profile_source",
    "query_database",
    "read_file_sample",
    "run_analysis_code",
]
