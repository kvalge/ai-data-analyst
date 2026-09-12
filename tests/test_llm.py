# test_llm.py

"""Tests for the Ollama wrapper: env models, primary default, think=false."""

from __future__ import annotations

import json
import socket
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from src.agent.llm import (
    ROLE_FALLBACK_FAST,
    LlmError,
    complete,
    model_for_role,
)
from src.config import Settings, load_settings

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Frozen settings with placeholder model names and a fake host."""
    return load_settings(
        environ={"OLLAMA_HOST": "http://ollama.test:11434", **_PLACEHOLDER_MODELS},
        load_dotenv_file=False,
        project_root=tmp_path,
    )


@pytest.fixture
def ollama_http(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the generate request. Isolation is fine: each test gets a new dict."""
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args: object):
            return False

        def read(self) -> bytes:
            return json.dumps({"response": "ok", "done": True}).encode()

    def fake_urlopen(request: Any, timeout: object = None) -> FakeResponse:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.agent.llm.urllib.request.urlopen", fake_urlopen)
    return captured


def test_complete_defaults_to_primary_model(
    settings: Settings, ollama_http: dict[str, Any]
):
    """Omitting role uses the primary model from settings."""
    complete("hello", settings=settings)
    assert ollama_http["body"]["model"] == settings.ollama_model_primary


def test_complete_uses_requested_role_model(
    settings: Settings, ollama_http: dict[str, Any]
):
    """An explicit role uses that role's env-configured name."""
    complete("hello", settings=settings, role=ROLE_FALLBACK_FAST)
    assert ollama_http["body"]["model"] == settings.ollama_model_fallback_fast


def test_complete_posts_to_settings_host(
    settings: Settings, ollama_http: dict[str, Any]
):
    """The generate URL is built from OLLAMA_HOST, not a hardcoded host."""
    complete("hello", settings=settings)
    assert ollama_http["url"] == "http://ollama.test:11434/api/generate"


def test_complete_uses_settings_timeout(
    settings: Settings, ollama_http: dict[str, Any]
):
    """The generate timeout is OLLAMA_TIMEOUT_S from settings, not a client literal."""
    complete("hello", settings=settings)
    assert ollama_http["timeout"] == settings.ollama_timeout_s


def test_structured_call_sets_think_false(
    settings: Settings, ollama_http: dict[str, Any]
):
    """Structured calls disable thinking so JSON is not wrapped in thoughts."""
    complete("hello", settings=settings, structured=True)
    assert ollama_http["body"]["think"] is False


def test_model_for_role_defaults_to_primary(settings: Settings):
    """The role picker itself defaults to primary."""
    assert model_for_role(settings) == settings.ollama_model_primary


def test_unknown_role_raises(settings: Settings):
    """An unknown role is a visible error, not a silent primary fallback."""
    with pytest.raises(LlmError, match="Unknown model role"):
        model_for_role(settings, "not-a-role")


def test_empty_model_for_role_raises(tmp_path):
    """A blank role name is not replaced with a hardcoded model."""
    settings = load_settings(
        environ={},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )
    with pytest.raises(LlmError, match="No model configured"):
        model_for_role(settings)


def test_cloud_model_is_rejected(tmp_path):
    """:cloud tags are off-machine and must not be called."""
    settings = load_settings(
        environ={
            **_PLACEHOLDER_MODELS,
            "OLLAMA_MODEL_PRIMARY": "placeholder-primary:cloud",
        },
        load_dotenv_file=False,
        project_root=tmp_path,
    )
    with pytest.raises(LlmError, match="Cloud Ollama"):
        model_for_role(settings)


def test_http_error_raises_llm_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    """A non-success HTTP status is surfaced, not swallowed."""

    def boom(request: object, timeout: object = None) -> None:
        raise HTTPError("http://ollama.test:11434/api/generate", 500, "err", None, BytesIO())

    monkeypatch.setattr("src.agent.llm.urllib.request.urlopen", boom)
    with pytest.raises(LlmError, match="Ollama HTTP 500"):
        complete("hello", settings=settings)


def test_unreachable_host_raises_llm_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    """A connection failure is a visible error, not a guessed reply."""

    def boom(request: object, timeout: object = None) -> None:
        raise URLError("connection refused")

    monkeypatch.setattr("src.agent.llm.urllib.request.urlopen", boom)
    with pytest.raises(LlmError, match="Could not reach Ollama"):
        complete("hello", settings=settings)


def test_timeout_raises_llm_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    """A stalled model load or generate is LlmError, not a raw socket.timeout."""

    def boom(request: object, timeout: object = None) -> None:
        raise socket.timeout("timed out")

    monkeypatch.setattr("src.agent.llm.urllib.request.urlopen", boom)
    with pytest.raises(LlmError, match="Could not reach Ollama"):
        complete("hello", settings=settings)
