# test_list_available_sources.py

"""Tests for list_available_sources and its tool contract."""

from datetime import datetime, timezone
from pathlib import Path

from src.storage.registry import save_file_source
from src.storage.sources import DataSource
from src.tools.list_sources import (
    LIST_AVAILABLE_SOURCES,
    list_available_sources,
    source_to_result,
)
from tests.tool_schema import assert_keys_match_required


def test_list_available_sources_empty(tmp_path: Path):
    """An empty upload dir yields an empty sources list."""
    assert list_available_sources(tmp_path) == {"sources": []}


def test_list_available_sources_one_file(tmp_path: Path, sample_sales_csv: Path):
    """A saved file source appears in the structured tool result."""
    upload_dir = tmp_path / "uploads"
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(incoming, upload_dir, original_name="sales.csv")

    result = list_available_sources(upload_dir)

    assert len(result["sources"]) == 1
    row = result["sources"][0]
    assert row["source_id"] == saved.source_id
    assert row["kind"] == "file"
    assert row["original_name"] == "sales.csv"
    assert row["stored_path"] == str(saved.stored_path)
    assert row["sha256"] == saved.sha256
    assert row["created_at"] == saved.created_at.isoformat()
    assert "date,region,revenue" not in str(result)


def test_list_available_sources_contract_is_complete():
    """The MCP-shaped contract has a name, description, and JSON schemas."""
    contract = LIST_AVAILABLE_SOURCES
    assert contract.name == "list_available_sources"
    assert contract.description
    assert contract.input_schema["type"] == "object"
    assert contract.result_schema["required"] == ["sources"]


def test_source_to_result_keys_match_result_schema():
    """Serialized keys stay aligned with the contract required list."""
    source = DataSource(
        source_id="file-abc",
        kind="file",
        original_name="sales.csv",
        stored_path=Path("sales.csv"),
        sha256="abc",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    items_schema = LIST_AVAILABLE_SOURCES.result_schema["properties"]["sources"][
        "items"
    ]
    assert_keys_match_required(source_to_result(source), items_schema)
