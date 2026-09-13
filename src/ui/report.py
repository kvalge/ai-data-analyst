# report.py

"""Downloadable markdown from the last tool summary and artifact paths."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import streamlit as st

_THREAD_PREFIX_LEN = 8


def last_tool_summary(result: Any) -> str:
    """Return the checkpointed one-line summary. Ignore every other key."""
    if not isinstance(result, dict):
        return ""
    text = result.get("summary")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""


def report_download_name(
    thread_id: str, *, when: datetime | None = None
) -> str:
    """Filename that distinguishes thread and download time."""
    prefix = "".join(ch for ch in thread_id.strip() if ch.isalnum())
    prefix = prefix[:_THREAD_PREFIX_LEN] or "thread"
    stamp = (when or datetime.now(timezone.utc)).astimezone(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    return f"report_{prefix}_{stamp}.md"


def build_report_markdown(
    *,
    summary: str,
    artifacts: Sequence[str],
) -> str:
    """Markdown from summary text and artifact paths. No row lists."""
    text = summary.strip()
    paths = [
        item.strip()
        for item in artifacts
        if isinstance(item, str) and item.strip()
    ]
    if not text and not paths:
        return ""
    parts = ["# Analysis report"]
    if text:
        parts.extend(["", "## Last summary", "", text])
    if paths:
        parts.extend(["", "## Artifacts", ""])
        parts.extend(f"- `{path}`" for path in paths)
    parts.append("")
    return "\n".join(parts)


def render_report_download(
    last_tool_result: Any,
    artifacts: Sequence[str],
    *,
    thread_id: str,
) -> None:
    """Offer report.md when a summary or artifact path exists."""
    markdown = build_report_markdown(
        summary=last_tool_summary(last_tool_result),
        artifacts=artifacts,
    )
    if not markdown:
        return
    st.download_button(
        "Download report.md",
        data=markdown,
        file_name=report_download_name(thread_id),
        mime="text/markdown",
        key="download_report_md",
    )
