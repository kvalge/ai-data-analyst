# audit.py

"""Append-only JSONL of tool use. Identities and timestamps only; no rows."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

AUDIT_LOG_NAME = "audit.jsonl"
_RECORD_KEYS = ("timestamp", "tool", "source_id")


def default_audit_dir(upload_dir: Path) -> Path:
    """Logs sit next to uploads (default `data/logs`). Not an env var."""
    return upload_dir.parent / "logs"


def audit_log_path(log_dir: Path) -> Path:
    """Return the append-only JSONL path inside `log_dir`."""
    return log_dir / AUDIT_LOG_NAME


def source_id_for_audit(arguments: dict[str, Any]) -> str | None:
    """Return a source or connection id when the tool named one. Do not invent."""
    for key in ("source_id", "connection_id"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def append_tool_use(
    log_dir: Path,
    *,
    tool: str,
    source_id: str | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append one JSON object. Never writes file contents or dataset rows."""
    # TODO: 8.4 catch mkdir/write failures (disk full, permissions) and log a
    # warning instead of raising after a successful tool result.
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = now if now is not None else datetime.now(timezone.utc)
    record = {
        "timestamp": stamp.isoformat(),
        "tool": tool,
        "source_id": source_id,
    }
    path = audit_log_path(log_dir)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    _LOG.info("audit tool=%s", tool)
    return {key: record[key] for key in _RECORD_KEYS}
