"""Tests for runtime directory creation."""

from src.config import load_settings
from src.storage.paths import ensure_runtime_dirs


def test_ensure_runtime_dirs_creates_expected_folders(tmp_path):
    """All runtime directories exist after the first call."""
    settings = load_settings(
        environ={},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )

    created = ensure_runtime_dirs(settings)

    assert settings.upload_dir.is_dir()
    assert settings.context_dir.is_dir()
    assert settings.cache_dir.is_dir()
    assert settings.artifact_dir.is_dir()
    assert settings.checkpoint_path.parent.is_dir()
    assert (tmp_path / "data" / "logs").is_dir()
    assert settings.upload_dir in created
    assert (tmp_path / "data" / "logs") in created
    assert len(created) == 6


def test_ensure_runtime_dirs_is_idempotent(tmp_path):
    """A second call does not fail and does not wipe existing files."""
    settings = load_settings(
        environ={},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )
    ensure_runtime_dirs(settings)
    marker = settings.upload_dir / "keep-me.txt"
    marker.write_text("ok", encoding="utf-8")

    ensure_runtime_dirs(settings)

    assert marker.is_file()
    assert marker.read_text(encoding="utf-8") == "ok"
    assert settings.upload_dir.is_dir()


def test_ensure_runtime_dirs_uses_explicit_log_dir(tmp_path):
    """An explicit log_dir is created instead of the default data/logs."""
    settings = load_settings(
        environ={},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )
    custom_logs = tmp_path / "custom-logs"

    ensure_runtime_dirs(settings, log_dir=custom_logs)

    assert custom_logs.is_dir()
    assert not (tmp_path / "data" / "logs").exists()


def test_ensure_runtime_dirs_dedupes_overlapping_paths(tmp_path):
    """Overlapping settings yield one entry per unique directory."""
    settings = load_settings(
        environ={
            "CHECKPOINT_PATH": str(tmp_path / "data" / "uploads" / "graph.sqlite"),
        },
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )

    created = ensure_runtime_dirs(settings)

    assert len(created) == len(set(created))
    assert created.count(settings.upload_dir) == 1
    assert settings.checkpoint_path.parent == settings.upload_dir
