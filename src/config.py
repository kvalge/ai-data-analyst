"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

# src/config.py → repository root. If this file moves, update the parents.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_UPLOAD_DIR = "./data/uploads"
DEFAULT_CONTEXT_DIR = "./data/context"
DEFAULT_CACHE_DIR = "./data/cache"
DEFAULT_ARTIFACT_DIR = "./data/artifacts"
DEFAULT_CHECKPOINT_PATH = "./data/checkpoints/graph.sqlite"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 5432
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_FULL_LOAD_ROWS = 100_000
DEFAULT_SAMPLE_N_ROWS = 50
DEFAULT_SANDBOX_TIMEOUT_S = 30
DEFAULT_MAX_PROMPT_CHARS = 8_000

_REDACTED_FIELDS = frozenset({"db_password"})


class SettingsError(ValueError):
    """Invalid or missing application settings."""


@dataclass(frozen=True)
class Settings:
    """Typed runtime settings. Model names come only from the environment."""

    ollama_host: str
    ollama_model_primary: str
    ollama_model_fallback_fast: str
    ollama_model_agentic: str
    ollama_model_coding: str
    upload_dir: Path
    context_dir: Path
    cache_dir: Path
    artifact_dir: Path
    checkpoint_path: Path
    max_upload_bytes: int
    max_full_load_rows: int
    sample_n_rows: int
    sandbox_timeout_s: int
    max_prompt_chars: int
    log_level: str
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str

    def __repr__(self) -> str:
        """Render settings without exposing secrets."""
        parts: list[str] = []
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in _REDACTED_FIELDS and value:
                value = "***"
            parts.append(f"{field.name}={value!r}")
        return f"Settings({', '.join(parts)})"


def load_settings(
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
    load_dotenv_file: bool = True,
    project_root: Path | None = None,
    require_models: bool = True,
) -> Settings:
    """Load settings from `environ` (or os.environ after optional .env).

    Model names have no safe default. App startup should keep
    `require_models=True` (the default) so a missing lineup fails immediately.
    Tests that only exercise paths or limits can pass `require_models=False`.
    """
    root = project_root or PROJECT_ROOT
    if environ is None:
        if load_dotenv_file:
            load_dotenv(dotenv_path or root / ".env")
        environ = os.environ

    settings = Settings(
        ollama_host=_get_str(environ, "OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        ollama_model_primary=_get_str(environ, "OLLAMA_MODEL_PRIMARY", ""),
        ollama_model_fallback_fast=_get_str(environ, "OLLAMA_MODEL_FALLBACK_FAST", ""),
        ollama_model_agentic=_get_str(environ, "OLLAMA_MODEL_AGENTIC", ""),
        ollama_model_coding=_get_str(environ, "OLLAMA_MODEL_CODING", ""),
        upload_dir=_get_path(environ, "UPLOAD_DIR", DEFAULT_UPLOAD_DIR, root),
        context_dir=_get_path(environ, "CONTEXT_DIR", DEFAULT_CONTEXT_DIR, root),
        cache_dir=_get_path(environ, "CACHE_DIR", DEFAULT_CACHE_DIR, root),
        artifact_dir=_get_path(environ, "ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR, root),
        checkpoint_path=_get_path(
            environ, "CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH, root
        ),
        max_upload_bytes=_get_int(
            environ, "MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES
        ),
        max_full_load_rows=_get_int(
            environ, "MAX_FULL_LOAD_ROWS", DEFAULT_MAX_FULL_LOAD_ROWS
        ),
        sample_n_rows=_get_int(environ, "SAMPLE_N_ROWS", DEFAULT_SAMPLE_N_ROWS),
        sandbox_timeout_s=_get_int(
            environ, "SANDBOX_TIMEOUT_S", DEFAULT_SANDBOX_TIMEOUT_S
        ),
        max_prompt_chars=_get_int(
            environ, "MAX_PROMPT_CHARS", DEFAULT_MAX_PROMPT_CHARS
        ),
        log_level=_get_str(environ, "LOG_LEVEL", DEFAULT_LOG_LEVEL),
        db_host=_get_str(environ, "DB_HOST", DEFAULT_DB_HOST),
        db_port=_get_int(environ, "DB_PORT", DEFAULT_DB_PORT),
        db_name=_get_str(environ, "DB_NAME", ""),
        db_user=_get_str(environ, "DB_USER", ""),
        db_password=_get_str(environ, "DB_PASSWORD", ""),
    )
    if require_models:
        _require_model_names(settings)
    return settings


def _require_model_names(settings: Settings) -> None:
    """Raise if any model role is unset. There is no hardcoded fallback name."""
    values = {
        "OLLAMA_MODEL_PRIMARY": settings.ollama_model_primary,
        "OLLAMA_MODEL_FALLBACK_FAST": settings.ollama_model_fallback_fast,
        "OLLAMA_MODEL_AGENTIC": settings.ollama_model_agentic,
        "OLLAMA_MODEL_CODING": settings.ollama_model_coding,
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise SettingsError(
            "Missing required settings: "
            + ", ".join(missing)
            + ". Set them in .env (see .env.example)."
        )


def _get_str(environ: Mapping[str, str], key: str, default: str) -> str:
    """Return a stripped env string, or default when missing/blank."""
    value = environ.get(key)
    if value is None or not value.strip():
        return default
    return value.strip()


def _get_int(environ: Mapping[str, str], key: str, default: int) -> int:
    """Return an env integer, or default when missing/blank."""
    value = environ.get(key)
    if value is None or not str(value).strip():
        return default
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise SettingsError(
            f"Environment variable {key} must be an integer, got {value!r}."
        ) from exc


def _get_path(
    environ: Mapping[str, str], key: str, default: str, project_root: Path
) -> Path:
    """Resolve an env path; relative values are under `project_root`."""
    raw = _get_str(environ, key, default)
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
