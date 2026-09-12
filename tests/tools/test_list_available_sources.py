# test_list_available_sources.py

"""Tests for list_available_sources and its tool contract."""

from datetime import datetime, timezone
from pathlib import Path

from src.config import load_settings
from src.storage.registry import save_file_source
from src.storage.sources import DataSource, make_postgres_source, postgres_source_id
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


def _db_settings(tmp_path: Path):
    return load_settings(
        environ={
            "DB_HOST": "db.local",
            "DB_PORT": "5432",
            "DB_NAME": "analytics",
            "DB_USER": "reader",
            "DB_PASSWORD": "super-secret-password",
        },
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )


def test_list_includes_postgres_when_enabled(tmp_path: Path, sample_sales_csv: Path):
    """A file plus the env DB appears when include_postgres is set."""
    upload_dir = tmp_path / "uploads"
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    save_file_source(incoming, upload_dir, original_name="sales.csv")
    settings = _db_settings(tmp_path)

    result = list_available_sources(
        upload_dir, settings=settings, include_postgres=True
    )

    kinds = [row["kind"] for row in result["sources"]]
    assert kinds == ["file", "postgres"]
    pg = result["sources"][1]
    assert pg["source_id"] == postgres_source_id(settings)
    assert pg["original_name"] == "reader@db.local:5432/analytics"
    assert pg["stored_path"] == ""
    assert "super-secret-password" not in str(result)


def test_list_omits_postgres_unless_enabled(tmp_path: Path):
    """Configured env is not a source until the caller opts in."""
    settings = _db_settings(tmp_path)
    assert list_available_sources(tmp_path, settings=settings) == {"sources": []}
    assert list_available_sources(
        tmp_path, settings=settings, include_postgres=False
    ) == {"sources": []}


def test_list_omits_postgres_when_not_configured(tmp_path: Path):
    """include_postgres does nothing without DB_NAME and DB_USER."""
    settings = load_settings(
        environ={},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )
    assert list_available_sources(
        tmp_path, settings=settings, include_postgres=True
    ) == {"sources": []}


def test_postgres_source_omits_password_and_is_stable(tmp_path: Path):
    """The same env settings produce the same id; password is not on the record."""
    settings = _db_settings(tmp_path)
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    source = make_postgres_source(settings, created_at=created)

    assert source.kind == "postgres"
    assert source.source_id.startswith("postgres-")
    assert source.stored_path is None
    assert "super-secret-password" not in repr(source)
    assert "super-secret-password" not in str(source_to_result(source))
    assert make_postgres_source(settings, created_at=created) == source
    other_password = load_settings(
        environ={
            "DB_HOST": "db.local",
            "DB_PORT": "5432",
            "DB_NAME": "analytics",
            "DB_USER": "reader",
            "DB_PASSWORD": "different-secret",
        },
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )
    assert postgres_source_id(other_password) == source.source_id
    assert_keys_match_required(
        source_to_result(source),
        LIST_AVAILABLE_SOURCES.result_schema["properties"]["sources"]["items"],
    )
