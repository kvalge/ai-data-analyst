# test_eval_corpus.py

"""Eval suite: corpus shape always, live-model checks under `pytest -m ollama`.

The corpus tests run in the default suite because a broken corpus would
silently weaken the eval. The generation tests need a local model, so they
carry the `ollama` marker and are skipped by default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.execute import ToolValidationError, interpret_model_reply
from src.agent.json_output import (
    FIRST_PARSE_FAILURE,
    RETRY_STRICT,
    JsonOutputError,
    decide_after_parse_failure,
    parse_json_output,
    retry_prompt_after_validation,
    strip_markdown_fences,
)
from src.agent.llm import complete
from src.config import Settings, SettingsError, load_settings
from src.execution.ast_check import PythonAstError, check_python_ast
from src.tools.registry import TOOL_REGISTRY
from tests.eval.corpus import (
    KIND_CODE,
    KIND_JSON_OBJECT,
    KIND_SPEED,
    KIND_TOOL_SELECTION,
    CorpusError,
    EvalCase,
    cases_of_kind,
    load_cases,
)

CASES = load_cases()
_IDS = [case.id for case in CASES]

_MINIMAL_CASE = (
    '{"id": "only", "kind": "speed", "role": "primary", '
    '"system_prompt": false, "user": "Say OK", "expect": {"max_words": 1}}'
)

# The graph retries once after these, so the eval must allow the same retry.
# A prose answer or the wrong tool name is not retried in production either.
_RETRYABLE = (JsonOutputError, ToolValidationError, PythonAstError)


@pytest.fixture(scope="module")
def eval_settings() -> Settings:
    """Settings with a real model lineup, or skip: the eval needs .env models."""
    try:
        return load_settings()
    except SettingsError as exc:
        pytest.skip(f"Ollama models are not configured: {exc}")


def _case(case_id: str) -> EvalCase:
    return next(case for case in CASES if case.id == case_id)


def test_corpus_covers_every_kind():
    """Each eval kind has at least one prompt, so no check silently disappears."""
    for kind in (KIND_TOOL_SELECTION, KIND_JSON_OBJECT, KIND_SPEED, KIND_CODE):
        assert cases_of_kind(kind, CASES), f"no {kind} case in the corpus"


def test_tool_selection_cases_name_registered_tools():
    """An expected tool must exist, or the assertion could never pass."""
    for case in cases_of_kind(KIND_TOOL_SELECTION, CASES):
        assert case.expect["tool"] in TOOL_REGISTRY


def test_tool_selection_covers_list_sample_and_profile():
    """The three tools that are easy to confuse are all represented."""
    expected = {
        case.expect["tool"] for case in cases_of_kind(KIND_TOOL_SELECTION, CASES)
    }
    assert {"list_available_sources", "read_file_sample", "profile_source"} <= expected


def test_tool_selection_prompt_carries_the_real_system_prompt():
    """Eval must exercise the shipped prompt, not a copy that can drift."""
    case = cases_of_kind(KIND_TOOL_SELECTION, CASES)[0]
    assert case.system_prompt is True
    assert "Registered tools" in case.prompt
    assert case.user in case.prompt


def test_code_case_runs_unstructured():
    """Per 4.7 the coding model is never asked for structured output."""
    for case in cases_of_kind(KIND_CODE, CASES):
        assert case.structured is False


def test_corpus_rejects_a_duplicate_id(tmp_path: Path):
    """A duplicate id would hide one case's result, so loading must fail."""
    path = tmp_path / "prompts.json"
    path.write_text(
        '{"cases": [' + ", ".join([_MINIMAL_CASE] * 2) + "]}",
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="Duplicate"):
        load_cases(path)


def test_corpus_rejects_an_unknown_kind(tmp_path: Path):
    """An unknown kind means no runner would check it; fail loudly instead."""
    path = tmp_path / "prompts.json"
    path.write_text(
        '{"cases": [' + _MINIMAL_CASE.replace('"speed"', '"guesswork"') + "]}",
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="Unknown eval kind"):
        load_cases(path)


def _generate(case: EvalCase, settings: Settings, prompt: str | None = None) -> str:
    """One call to the case's role, with the case's structured flag."""
    return complete(
        prompt if prompt is not None else case.prompt,
        settings=settings,
        role=case.role,
        structured=case.structured,
    )


def _check_reply(case: EvalCase, reply: str) -> None:
    """Assert the reply satisfies the case. Retryable failures raise, not assert."""
    assert reply.strip(), f"{case.id}: model returned nothing"
    if case.kind == KIND_TOOL_SELECTION:
        call = interpret_model_reply(reply)
        # None means prose; the graph answers the user instead of retrying.
        assert call is not None, f"{case.id}: expected a tool call, got prose"
        assert call["name"] == case.expect["tool"], (
            f"{case.id}: chose {call['name']}, expected {case.expect['tool']}"
        )
        return
    if case.kind == KIND_JSON_OBJECT:
        schema = {
            "type": "object",
            "properties": {key: {} for key in case.expect["required_keys"]},
            "required": list(case.expect["required_keys"]),
        }
        # Fenced output must survive stripping; that is the trap in this case.
        assert parse_json_output(reply, schema)
        return
    if case.kind == KIND_SPEED:
        assert len(reply.split()) <= case.expect["max_words"]
        return
    code = strip_markdown_fences(reply)
    for needle in case.expect["must_include"]:
        assert needle in code, f"{case.id}: {needle!r} missing from generated code"
    if case.expect.get("ast_check"):
        check_python_ast(code)


@pytest.mark.ollama
@pytest.mark.parametrize("case_id", _IDS)
def test_model_output_parses(case_id: str, eval_settings: Settings):
    """Corpus prompts pass our parsers within the shipped one-retry budget."""
    case = _case(case_id)
    try:
        _check_reply(case, _generate(case, eval_settings))
        return
    except _RETRYABLE as exc:
        assert decide_after_parse_failure(FIRST_PARSE_FAILURE) == RETRY_STRICT
        stricter = retry_prompt_after_validation(
            case.prompt, str(exc), limit=eval_settings.max_prompt_chars
        )
    _check_reply(case, _generate(case, eval_settings, stricter))
