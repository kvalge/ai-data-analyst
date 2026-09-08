# list_sources.py

"""Tool: list registered analysis data sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import Settings
from src.db.postgres import postgres_configured
from src.storage.registry import list_file_sources
from src.storage.sources import DataSource, make_postgres_source
from src.tools.contracts import ToolContract, empty_object_schema

LIST_AVAILABLE_SOURCES = ToolContract(
    name="list_available_sources",
    description=(
        "List analysis data sources currently uploaded or registered. "
        "Returns source_id, kind, original_name, stored_path, and sha256. "
        "May include one Postgres source from env when the user enables it. "
        "Does not return file contents or the database password."
    ),
    input_schema=empty_object_schema(),
    result_schema={
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "kind": {"type": "string"},
                        "original_name": {"type": "string"},
                        "stored_path": {"type": "string"},
                        "sha256": {"type": "string"},
                        "created_at": {"type": "string"},
                    },
                    "required": [
                        "source_id",
                        "kind",
                        "original_name",
                        "stored_path",
                        "sha256",
                        "created_at",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["sources"],
        "additionalProperties": False,
    },
)


def source_to_result(source: DataSource) -> dict[str, str]:
    """Serialize a DataSource for a structured tool result (no file bytes).

    Keep this as a function while only one tool needs it. If a second caller
    appears, move it to DataSource.to_dict().
    """
    stored = "" if source.stored_path is None else str(source.stored_path)
    return {
        "source_id": source.source_id,
        "kind": source.kind,
        "original_name": source.original_name,
        "stored_path": stored,
        "sha256": source.sha256,
        "created_at": source.created_at.isoformat(),
    }


def list_available_sources(
    upload_dir: Path,
    *,
    settings: Settings | None = None,
    include_postgres: bool = False,
) -> dict[str, Any]:
    """Return file sources, and optionally one env-backed Postgres source.

    `upload_dir`, `settings`, and `include_postgres` are injected by the app,
    not the LLM. The password is never included.
    """
    sources = [source_to_result(item) for item in list_file_sources(upload_dir)]
    if (
        include_postgres
        and settings is not None
        and postgres_configured(settings)
    ):
        sources.append(source_to_result(make_postgres_source(settings)))
    return {"sources": sources}
