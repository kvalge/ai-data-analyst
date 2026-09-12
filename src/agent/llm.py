# llm.py

"""Local Ollama client. Host and model names come from settings, never literals."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from src.config import Settings

_LOG = logging.getLogger(__name__)

ROLE_PRIMARY = "primary"
ROLE_FALLBACK_FAST = "fallback_fast"
ROLE_AGENTIC = "agentic"
ROLE_CODING = "coding"

_ROLE_FIELDS: dict[str, str] = {
    ROLE_PRIMARY: "ollama_model_primary",
    ROLE_FALLBACK_FAST: "ollama_model_fallback_fast",
    ROLE_AGENTIC: "ollama_model_agentic",
    ROLE_CODING: "ollama_model_coding",
}


class LlmError(ValueError):
    """Ollama could not complete the request. The caller must not guess a reply."""


def model_for_role(settings: Settings, role: str = ROLE_PRIMARY) -> str:
    """Return the env-configured model for `role`. Default role is primary."""
    field = _ROLE_FIELDS.get(role)
    if field is None:
        raise LlmError(f"Unknown model role: {role}.")
    model = getattr(settings, field)
    if not isinstance(model, str) or not model.strip():
        raise LlmError(f"No model configured for role {role}.")
    model = model.strip()
    if ":cloud" in model.lower():
        raise LlmError("Cloud Ollama models are not allowed.")
    return model


def complete(
    prompt: str,
    *,
    settings: Settings,
    role: str = ROLE_PRIMARY,
    structured: bool = True,
    timeout_s: int | None = None,
) -> str:
    """POST /api/generate and return the text. Structured calls set think=false."""
    model = model_for_role(settings, role)
    seconds = settings.ollama_timeout_s if timeout_s is None else timeout_s
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if structured:
        payload["think"] = False
        payload["options"] = {"temperature": 0}
    body = _post_generate(settings.ollama_host, payload, seconds)
    text = body.get("response")
    if not isinstance(text, str):
        raise LlmError("Ollama response was missing.")
    _LOG.info("ollama complete role=%s model=%s", role, model)
    return text


def _post_generate(host: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    """Send one non-streaming generate request. Fail visibly on transport errors."""
    request = urllib.request.Request(
        url=f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        _LOG.info("ollama http error")
        raise LlmError(f"Ollama HTTP {exc.code}.") from exc
    except OSError as exc:
        # URLError, socket.timeout / TimeoutError, connection refused, stalled read.
        _LOG.info("ollama connection error")
        raise LlmError("Could not reach Ollama.") from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LlmError("Ollama returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise LlmError("Ollama returned invalid JSON.")
    return parsed
