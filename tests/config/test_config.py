# test_config.py

"""Tests for env-backed application settings."""

import pytest

from src.config import (
    DEFAULT_MAX_FULL_LOAD_ROWS,
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_MAX_PROMPT_TURNS,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_TIMEOUT_S,
    DEFAULT_SAMPLE_N_ROWS,
    DEFAULT_SANDBOX_TIMEOUT_S,
    SettingsError,
    load_settings,
)

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}


def test_defaults_resolve(tmp_path):
    """Missing optional env vars fall back to documented defaults."""
    settings = load_settings(
        environ={},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )

    assert settings.ollama_host == DEFAULT_OLLAMA_HOST
    assert settings.ollama_model_primary == ""
    assert settings.ollama_model_fallback_fast == ""
    assert settings.ollama_model_agentic == ""
    assert settings.ollama_model_coding == ""
    assert settings.upload_dir == (tmp_path / "data" / "uploads").resolve()
    assert settings.context_dir == (tmp_path / "data" / "context").resolve()
    assert settings.cache_dir == (tmp_path / "data" / "cache").resolve()
    assert settings.artifact_dir == (tmp_path / "data" / "artifacts").resolve()
    assert settings.checkpoint_path == (
        tmp_path / "data" / "checkpoints" / "graph.sqlite"
    ).resolve()
    assert settings.max_upload_bytes == DEFAULT_MAX_UPLOAD_BYTES
    assert settings.max_full_load_rows == DEFAULT_MAX_FULL_LOAD_ROWS
    assert settings.sample_n_rows == DEFAULT_SAMPLE_N_ROWS
    assert settings.sandbox_timeout_s == DEFAULT_SANDBOX_TIMEOUT_S
    assert settings.ollama_timeout_s == DEFAULT_OLLAMA_TIMEOUT_S
    assert settings.max_prompt_chars == DEFAULT_MAX_PROMPT_CHARS
    assert settings.max_prompt_turns == DEFAULT_MAX_PROMPT_TURNS
    assert settings.log_level == "INFO"


def test_max_prompt_turns_from_env(tmp_path):
    """MAX_PROMPT_TURNS is read from the environment."""
    settings = load_settings(
        environ={"MAX_PROMPT_TURNS": "2"},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )
    assert settings.max_prompt_turns == 2


def test_ollama_timeout_from_env(tmp_path):
    """OLLAMA_TIMEOUT_S is read from the environment."""
    settings = load_settings(
        environ={"OLLAMA_TIMEOUT_S": "90"},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )

    assert settings.ollama_timeout_s == 90


def test_upload_dir_from_env(tmp_path):
    """UPLOAD_DIR is read from the environment and resolved."""
    custom = tmp_path / "custom_uploads"
    settings = load_settings(
        environ={"UPLOAD_DIR": str(custom)},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )

    assert settings.upload_dir == custom.resolve()


def test_model_names_come_from_env(tmp_path):
    """Model names are whatever the environment provides, not hardcoded."""
    settings = load_settings(
        environ=_PLACEHOLDER_MODELS,
        load_dotenv_file=False,
        project_root=tmp_path,
    )

    assert settings.ollama_model_primary == "placeholder-primary:tag"
    assert settings.ollama_model_fallback_fast == "placeholder-fast:tag"
    assert settings.ollama_model_agentic == "placeholder-agentic:tag"
    assert settings.ollama_model_coding == "placeholder-coding:tag"


def test_missing_models_raise_at_startup(tmp_path):
    """App-style load (require_models=True) fails with the missing env names."""
    with pytest.raises(SettingsError, match="OLLAMA_MODEL_PRIMARY") as exc_info:
        load_settings(
            environ={},
            load_dotenv_file=False,
            project_root=tmp_path,
        )

    message = str(exc_info.value)
    assert "OLLAMA_MODEL_FALLBACK_FAST" in message
    assert "OLLAMA_MODEL_AGENTIC" in message
    assert "OLLAMA_MODEL_CODING" in message


def test_invalid_int_names_the_variable(tmp_path):
    """A bad integer setting names the environment variable."""
    with pytest.raises(SettingsError, match="MAX_UPLOAD_BYTES") as exc_info:
        load_settings(
            environ={"MAX_UPLOAD_BYTES": "not-a-number"},
            load_dotenv_file=False,
            project_root=tmp_path,
            require_models=False,
        )

    assert "not-a-number" in str(exc_info.value)


def test_repr_redacts_db_password(tmp_path):
    """Settings repr must not leak the database password."""
    settings = load_settings(
        environ={"DB_PASSWORD": "super-secret-password"},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )

    rendered = repr(settings)
    assert "super-secret-password" not in rendered
    assert "db_password='***'" in rendered


def test_repr_shows_empty_db_password_when_unset(tmp_path):
    """Unset password is empty in repr, not masked, so 'not set' stays visible."""
    settings = load_settings(
        environ={},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )

    assert settings.db_password == ""
    assert "db_password=''" in repr(settings)
