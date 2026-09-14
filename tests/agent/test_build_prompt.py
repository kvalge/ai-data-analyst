# test_build_prompt.py

"""Tests that the LLM prompt lists prior summaries and artifact paths."""

from src.agent.graph import build_prompt
from src.agent.state import empty_agent_state
from src.config import DEFAULT_MAX_ARTIFACTS, DEFAULT_MAX_PROMPT_TURNS


def test_build_prompt_lists_artifact_paths():
    """Follow-up turns see stored paths, not file contents."""
    state = empty_agent_state()
    state["messages"] = [{"role": "user", "content": "break that down by region"}]
    state["artifacts"] = [r"C:\data\artifacts\summary.csv"]
    text = build_prompt(state, max_turns=DEFAULT_MAX_PROMPT_TURNS)
    assert r"C:\data\artifacts\summary.csv" in text
    assert "Available artifacts" in text
    assert "date,region,revenue" not in text


def test_build_prompt_omits_artifacts_when_none():
    """An empty artifact list does not add a blank section."""
    state = empty_agent_state()
    state["messages"] = [{"role": "user", "content": "hello"}]
    text = build_prompt(state, max_turns=DEFAULT_MAX_PROMPT_TURNS)
    assert "Available artifacts" not in text


def test_build_prompt_lists_only_last_artifacts():
    """A long stored list is capped in the prompt, same as MAX_PROMPT_TURNS."""
    state = empty_agent_state()
    state["messages"] = [{"role": "user", "content": "again"}]
    state["artifacts"] = [
        f"C:/data/artifacts/{index}.csv"
        for index in range(DEFAULT_MAX_ARTIFACTS + 1)
    ]
    text = build_prompt(state, max_turns=DEFAULT_MAX_PROMPT_TURNS)
    assert "C:/data/artifacts/0.csv" not in text
    assert f"C:/data/artifacts/{DEFAULT_MAX_ARTIFACTS}.csv" in text


def test_build_prompt_includes_retrieved_snippet():
    """A retrieve result injects snippet text into the next prompt."""
    state = empty_agent_state()
    state["messages"] = [{"role": "user", "content": "what is revenue?"}]
    state["last_tool_result"] = {
        "name": "retrieve_domain_context",
        "summary": "retrieve_domain_context: 1 chunk(s)",
        "chunks": [
            {
                "source_file": "sample_glossary.md",
                "chunk_id": "sample_glossary.md:0",
                "text": "Revenue is net of returns.",
                "score": 0.5,
            }
        ],
    }
    text = build_prompt(state, max_turns=DEFAULT_MAX_PROMPT_TURNS)
    assert "Revenue is net of returns." in text
    assert "date,region,revenue" not in text


def test_build_prompt_includes_last_tool_summary():
    """The latest compacted tool result stays in the prompt."""
    state = empty_agent_state()
    state["messages"] = [{"role": "user", "content": "again"}]
    state["last_tool_result"] = {
        "name": "run_analysis_code",
        "summary": "run_analysis_code: 1 artifact(s)",
    }
    text = build_prompt(state, max_turns=DEFAULT_MAX_PROMPT_TURNS)
    assert "run_analysis_code: 1 artifact(s)" in text
    assert "Last tool result" in text
