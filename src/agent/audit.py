# audit.py

"""Append-only JSONL of tool use. Identities, and code-tool HITL/outcome; no rows."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.code_approval import generated_code_text

_LOG = logging.getLogger(__name__)

AUDIT_LOG_NAME = "audit.jsonl"
_RECORD_KEYS = ("timestamp", "tool", "source_id")
_CODE_RECORD_KEYS = ("timestamp", "tool", "source_id", "code", "decision", "outcome")

DECISION_AUTO = "auto"
OUTCOME_SUCCESS = "success"
OUTCOME_ERROR = "error"
OUTCOME_REJECTED = "rejected"


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


def code_text_for_audit(
    name: str, arguments: Mapping[str, Any], limit: int
) -> str:
    """SQL or Python from the tool args, capped. Never dataset rows."""
    text = generated_code_text(name, arguments)
    if limit < 1:
        return text
    return text[:limit]


def append_tool_use(
    log_dir: Path,
    *,
    tool: str,
    source_id: str | None,
    now: datetime | None = None,
    code: str | None = None,
    decision: str | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    """Append one JSON object. Never writes file contents or dataset rows."""
    # TODO: 8.4 catch mkdir/write failures (disk full, permissions) and log a
    # warning instead of raising after a successful tool result.
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = now if now is not None else datetime.now(timezone.utc)
    record: dict[str, Any] = {
        "timestamp": stamp.isoformat(),
        "tool": tool,
        "source_id": source_id,
    }
    keys = _RECORD_KEYS
    if code is not None or decision is not None or outcome is not None:
        record["code"] = code
        record["decision"] = decision
        record["outcome"] = outcome
        keys = _CODE_RECORD_KEYS
    path = audit_log_path(log_dir)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    _LOG.info("audit tool=%s", tool)
    return {key: record[key] for key in keys}
