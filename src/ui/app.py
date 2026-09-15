# app.py
# streamlit run src/ui/app.py

"""Streamlit shell: HITL mode, uploads, preview, profile, chat, and Postgres."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# src/ui/app.py → repository root. Streamlit puts this file's dir on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.config import load_settings
from src.db.postgres import DatabaseError, check_postgres_connection, postgres_configured
from src.logging_setup import configure_logging
from src.rag.index import reindex_context
from src.rag.readers import ContextReadError
from src.storage.context import ingest_context_upload, list_context_files
from src.storage.ingest import ingest_data_upload
from src.storage.paths import ensure_runtime_dirs
from src.storage.registry import RegistryError
from src.tools.list_sources import list_available_sources
from src.tools.profile_source import ProfileError, profile_source
from src.tools.read_sample import read_file_sample
from src.ui.chat import render_chat, start_new_chat
from src.ui.hitl import HITL_MODE_KEY, HitlMode, ensure_hitl_mode
from src.ui.profile import (
    PROFILE_SOURCE_ID_KEY,
    SELECTED_SOURCE_ID_KEY,
    render_closable_heading,
    render_profile,
)
from src.validation.context_files import CONTEXT_FILE_SUFFIXES
from src.validation.data_files import DATA_FILE_SUFFIXES
from src.validation.uploads import FileValidationError

PREVIEW_SOURCE_ID_KEY = "preview_source_id"

settings = load_settings(require_models=False)
configure_logging(settings.log_level)
ensure_runtime_dirs(settings)
logging.getLogger(__name__).info("UI started")

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("AI Data Analyst")

data_types = sorted(suffix.lstrip(".") for suffix in DATA_FILE_SUFFIXES)
context_types = sorted(suffix.lstrip(".") for suffix in CONTEXT_FILE_SUFFIXES)

with st.sidebar:
    st.header("Session")
    ensure_hitl_mode(st.session_state)
    st.radio("HITL mode", options=list(HitlMode), key=HITL_MODE_KEY)
    if st.button("New chat", key="new_chat"):
        start_new_chat(
            st.session_state, checkpoint_path=settings.checkpoint_path
        )
        st.rerun()

    st.subheader("Postgres")
    if postgres_configured(settings):
        st.caption(
            f"{settings.db_user}@{settings.db_host}:{settings.db_port}/"
            f"{settings.db_name}"
        )
        st.caption("The database role must be read-only (enforced in Postgres, not here).")
        st.checkbox("Use configured database", key="use_configured_db")
        if st.button("Test connection", key="postgres_test"):
            try:
                check_postgres_connection(settings)
                st.session_state.postgres_test_ok = True
                st.session_state.postgres_test_error = None
            except DatabaseError as exc:
                st.session_state.postgres_test_ok = False
                st.session_state.postgres_test_error = str(exc)
        if st.session_state.get("postgres_test_ok"):
            st.success("Postgres connection ok.")
        elif st.session_state.get("postgres_test_error"):
            st.error(st.session_state.postgres_test_error)
    else:
        st.caption("No database configured (set DB_NAME and DB_USER).")

    uploaded = st.file_uploader("Data file", type=data_types, key="data_file")
    if uploaded is not None:
        payload = uploaded.getvalue()
        token = f"{uploaded.name}:{len(payload)}"
        if st.session_state.get("ingested_data_token") != token:
            try:
                source = ingest_data_upload(
                    payload,
                    uploaded.name,
                    settings.upload_dir,
                    max_bytes=settings.max_upload_bytes,
                )
                # A fresh upload selects itself, so chat does not stay on an
                # older source. Later reruns keep whatever the user picks.
                st.session_state[SELECTED_SOURCE_ID_KEY] = source.source_id
                st.session_state.ingested_data_token = token
                st.session_state.data_upload_error = None
            except (FileValidationError, RegistryError) as exc:
                st.session_state.data_upload_error = str(exc)
        if st.session_state.get("data_upload_error"):
            st.error(st.session_state.data_upload_error)

    context_uploaded = st.file_uploader(
        "Context file", type=context_types, key="context_file"
    )
    st.caption("Same filename replaces the previous context file.")
    if context_uploaded is not None:
        payload = context_uploaded.getvalue()
        token = f"{context_uploaded.name}:{len(payload)}"
        if st.session_state.get("ingested_context_token") != token:
            try:
                ingest_context_upload(
                    payload,
                    context_uploaded.name,
                    settings.context_dir,
                    max_bytes=settings.max_upload_bytes,
                )
                reindex_context(
                    settings.context_dir,
                    cache_dir=settings.cache_dir,
                    max_bytes=settings.max_upload_bytes,
                )
                st.session_state.ingested_context_token = token
                st.session_state.context_upload_error = None
            except (FileValidationError, ContextReadError) as exc:
                st.session_state.context_upload_error = str(exc)
        if st.session_state.get("context_upload_error"):
            st.error(st.session_state.context_upload_error)

    st.subheader("Data sources")
    listed = list_available_sources(
        settings.upload_dir,
        settings=settings,
        include_postgres=bool(st.session_state.get("use_configured_db")),
    )
    if not listed["sources"]:
        st.caption("No data sources yet.")
    labels = {
        row["source_id"]: f"{row['original_name']} ({row['kind']})"
        for row in listed["sources"]
    }
    for row in listed["sources"]:
        st.write(f"{row['original_name']} ({row['kind']}, `{row['source_id'][:20]}…`)")
        if row["kind"] == "file" and st.button(
            "Preview", key=f"preview-{row['source_id']}"
        ):
            st.session_state[PREVIEW_SOURCE_ID_KEY] = row["source_id"]
    if listed["sources"]:
        options = [row["source_id"] for row in listed["sources"]]
        current = st.session_state.get(SELECTED_SOURCE_ID_KEY)
        if current not in options:
            st.session_state[SELECTED_SOURCE_ID_KEY] = options[0]
        st.selectbox(
            "Selected source",
            options=options,
            format_func=lambda source_id: labels[source_id],
            key=SELECTED_SOURCE_ID_KEY,
        )
        if st.button("Profile selected source"):
            st.session_state[PROFILE_SOURCE_ID_KEY] = st.session_state[
                SELECTED_SOURCE_ID_KEY
            ]
            logging.getLogger(__name__).info(
                "UI profile requested source_id=%s",
                st.session_state[PROFILE_SOURCE_ID_KEY],
            )

    st.subheader("Context files")
    context_listed = list_context_files(settings.context_dir)
    if not context_listed:
        st.caption("No context files yet.")
    for path in context_listed:
        st.write(path.name)

profile_id = st.session_state.get(PROFILE_SOURCE_ID_KEY)
if profile_id:
    if render_closable_heading("Profile", close_key="close_profile"):
        st.session_state.pop(PROFILE_SOURCE_ID_KEY, None)
        st.rerun()
    try:
        result = profile_source(
            upload_dir=settings.upload_dir,
            cache_dir=settings.cache_dir,
            n_rows=settings.sample_n_rows,
            max_bytes=settings.max_upload_bytes,
            source_id=profile_id,
            settings=settings,
        )
        render_profile(result)
    except (FileValidationError, ProfileError, RegistryError) as exc:
        st.error(str(exc))

preview_id = st.session_state.get(PREVIEW_SOURCE_ID_KEY)
if preview_id:
    if render_closable_heading("Preview", close_key="close_preview"):
        st.session_state.pop(PREVIEW_SOURCE_ID_KEY, None)
        st.rerun()
    try:
        sample = read_file_sample(
            upload_dir=settings.upload_dir,
            n_rows=settings.sample_n_rows,
            max_bytes=settings.max_upload_bytes,
            source_id=preview_id,
        )
        st.caption(
            f"{sample['row_count']} row(s) · " + ", ".join(sample["columns"])
        )
        st.dataframe(sample["rows"], hide_index=True)
    except (FileValidationError, RegistryError) as exc:
        st.error(str(exc))

if not profile_id and not preview_id:
    st.info(
        "Preview and profile can both stay open; close either from its panel. "
        "Chat is below. The model may list, sample, profile, or load a file."
    )

st.subheader("Chat")
selected = st.session_state.get(SELECTED_SOURCE_ID_KEY)
active_source = labels.get(selected) if isinstance(selected, str) else None
st.caption(
    "Local Ollama only. The model may call list, sample, profile, or load. "
    + (
        f"Chat uses **{active_source}**; change it in the sidebar."
        if active_source
        else "No source selected."
    )
)
render_chat(
    settings,
    source_ids=[selected] if isinstance(selected, str) and selected else None,
)
