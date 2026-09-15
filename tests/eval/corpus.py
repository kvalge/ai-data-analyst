# corpus.py

"""Load the shared eval prompt corpus.

Both the pytest eval suite and `scripts/compare_ollama_models.py` read the
corpus through this module, so a prompt change lands in both at once.
Prompts are synthetic: no dataset rows and no credentials.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.llm import (
    ROLE_AGENTIC,
    ROLE_CODING,
    ROLE_FALLBACK_FAST,
    ROLE_PRIMARY,
)
from src.agent.prompts import build_system_prompt

CORPUS_PATH = Path(__file__).resolve().parent / "prompts.json"

KIND_TOOL_SELECTION = "tool_selection"
KIND_JSON_OBJECT = "json_object"
KIND_SPEED = "speed"
KIND_CODE = "code"

ALLOWED_KINDS = frozenset(
    {KIND_TOOL_SELECTION, KIND_JSON_OBJECT, KIND_SPEED, KIND_CODE}
)
ALLOWED_ROLES = frozenset(
    {ROLE_PRIMARY, ROLE_FALLBACK_FAST, ROLE_AGENTIC, ROLE_CODING}
)

_REQUIRED_FIELDS = ("id", "kind", "role", "system_prompt", "user", "expect")


class CorpusError(ValueError):
    """The corpus file is missing or malformed. Do not guess a prompt."""


@dataclass(frozen=True)
class EvalCase:
    """One prompt plus what a correct answer must satisfy."""

    id: str
    kind: str
    role: str
    system_prompt: bool
    user: str
    expect: dict[str, Any]

    @property
    def prompt(self) -> str:
        """Text sent to the model, with the real system prompt when requested."""
        if not self.system_prompt:
            return self.user
        # Same shape as the graph's prompt: system text, then the user turn.
        return f"{build_system_prompt()}\n\nuser: {self.user}"

    @property
    def structured(self) -> bool:
        """Code generation runs unstructured; everything else sets think=false."""
        return self.kind != KIND_CODE


def load_cases(path: Path = CORPUS_PATH) -> list[EvalCase]:
    """Read and validate the corpus. A malformed corpus raises, never skips."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusError(f"Could not read the eval corpus: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusError(f"Eval corpus is not valid JSON: {path}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise CorpusError("Eval corpus must be an object with a cases list.")
    cases = [_build_case(entry) for entry in raw["cases"]]
    if not cases:
        raise CorpusError("Eval corpus has no cases.")
    _require_unique_ids(cases)
    return cases


def cases_of_kind(kind: str, cases: list[EvalCase] | None = None) -> list[EvalCase]:
    """Filter the corpus by kind, e.g. only the one-word speed prompts."""
    if kind not in ALLOWED_KINDS:
        raise CorpusError(f"Unknown eval kind: {kind}")
    pool = cases if cases is not None else load_cases()
    return [case for case in pool if case.kind == kind]


def _build_case(entry: object) -> EvalCase:
    """Turn one corpus entry into a validated case."""
    if not isinstance(entry, dict):
        raise CorpusError("Each eval case must be an object.")
    missing = [field for field in _REQUIRED_FIELDS if field not in entry]
    if missing:
        raise CorpusError("Eval case is missing keys: " + ", ".join(missing))
    kind = str(entry["kind"])
    if kind not in ALLOWED_KINDS:
        raise CorpusError(f"Unknown eval kind: {kind}")
    role = str(entry["role"])
    if role not in ALLOWED_ROLES:
        raise CorpusError(f"Unknown model role: {role}")
    user = str(entry["user"]).strip()
    if not user:
        raise CorpusError(f"Eval case {entry['id']} has empty user text.")
    if not isinstance(entry["expect"], dict) or not entry["expect"]:
        raise CorpusError(f"Eval case {entry['id']} needs a non-empty expect.")
    if not isinstance(entry["system_prompt"], bool):
        raise CorpusError(f"Eval case {entry['id']} system_prompt must be a bool.")
    return EvalCase(
        id=str(entry["id"]),
        kind=kind,
        role=role,
        system_prompt=entry["system_prompt"],
        user=user,
        expect=dict(entry["expect"]),
    )


def _require_unique_ids(cases: list[EvalCase]) -> None:
    """Ids name the test case, so a duplicate would hide a result."""
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise CorpusError(f"Duplicate eval case id: {case.id}")
        seen.add(case.id)
