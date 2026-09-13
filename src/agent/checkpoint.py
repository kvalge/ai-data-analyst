# checkpoint.py

"""Sqlite checkpointer and the current thread id beside CHECKPOINT_PATH."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

_LOG = logging.getLogger(__name__)


def thread_id_path(checkpoint_path: Path) -> Path:
    """Sidecar file that remembers the current thread across process restarts."""
    return checkpoint_path.with_suffix(".thread")


def load_persisted_thread_id(checkpoint_path: Path) -> str | None:
    """Return the stored thread id, or None if it is missing or blank."""
    path = thread_id_path(checkpoint_path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def persist_thread_id(checkpoint_path: Path, thread_id: str) -> None:
    """Write the current thread id next to the sqlite file."""
    path = thread_id_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(thread_id.strip() + "\n", encoding="utf-8")
    _LOG.info("checkpoint thread persisted")


def sqlite_checkpointer(checkpoint_path: Path) -> SqliteSaver:
    """Open a process-lived SqliteSaver. Do not close the connection.

    ``SqliteSaver.from_conn_string`` closes the file when the context exits.
    Streamlit keeps the compiled graph in session_state, so the connection
    must stay open. ``check_same_thread=False`` allows Streamlit reruns.
    WAL lets a second browser tab read while another session writes; a short
    busy timeout waits out a colliding writer instead of raising immediately.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    saver = SqliteSaver(conn)
    saver.setup()
    _LOG.info("checkpoint sqlite path=%s", checkpoint_path)
    return saver
