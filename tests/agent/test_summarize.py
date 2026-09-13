# test_summarize.py

"""Tests for compact tool results: summary + paths, not row lists."""

from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

from src.agent.summarize import (
    artifact_paths_from_result,
    checkpoint_tool_result,
    is_json_safe,
    merge_artifacts,
    summarize_tool_result,
)


def test_is_json_safe_accepts_plain_dict():
    """A summary-shaped payload can sit in a checkpoint."""
    assert is_json_safe({"name": "list_available_sources", "sources": []})


def test_is_json_safe_rejects_dataframe():
    """A DataFrame must not be stored; json.dumps without default fails closed."""
    frame = pd.DataFrame({"region": ["North"], "revenue": [120]})
    assert is_json_safe(frame) is False
    assert is_json_safe({"last_tool_result": frame}) is False


def test_summarize_drops_rows_and_adds_summary():
    """Row records leave the stored result. Metadata and a summary stay."""
    stored = summarize_tool_result(
        "read_file_sample",
        {
            "columns": ["date", "region", "revenue"],
            "row_count": 3,
            "rows": [{"region": "North", "revenue": 120}],
        },
    )
    assert "rows" not in stored
    assert stored["columns"] == ["date", "region", "revenue"]
    assert stored["name"] == "read_file_sample"
    assert "3 rows" in stored["summary"]
    assert "North" not in stored["summary"]
    assert json.dumps(stored) == json.dumps(json.loads(json.dumps(stored)))


def test_artifact_paths_from_result_keeps_strings_only():
    """Only non-empty path strings are collected."""
    assert artifact_paths_from_result(
        {"artifacts": ["C:/data/artifacts/a.csv", "", 1, None]}
    ) == ["C:/data/artifacts/a.csv"]


def test_checkpoint_appends_new_artifact_paths():
    """A later tool adds paths; the previous path is kept."""
    updates = checkpoint_tool_result(
        name="run_analysis_code",
        result={
            "source_id": "file-abc",
            "stdout": "ok",
            "artifacts": ["C:/data/artifacts/chart.png"],
        },
        prior_artifacts=["C:/data/artifacts/old.csv"],
    )
    assert updates["artifacts"] == [
        "C:/data/artifacts/old.csv",
        "C:/data/artifacts/chart.png",
    ]
    assert "1 artifact(s)" in updates["last_tool_result"]["summary"]
    assert updates["error"] is None


def test_merge_artifacts_dedups_repeat_path():
    """The same path is stored once; a repeat becomes the newest entry."""
    assert merge_artifacts(
        ["C:/data/artifacts/a.csv", "C:/data/artifacts/b.csv"],
        ["C:/data/artifacts/a.csv"],
    ) == ["C:/data/artifacts/b.csv", "C:/data/artifacts/a.csv"]


def test_merge_artifacts_keeps_last_k():
    """Older paths drop when the list exceeds the cap."""
    prior = [f"C:/data/artifacts/{index}.csv" for index in range(8)]
    merged = merge_artifacts(
        prior, ["C:/data/artifacts/new.csv"], max_artifacts=8
    )
    assert "C:/data/artifacts/0.csv" not in merged
    assert merged[-1] == "C:/data/artifacts/new.csv"
    assert len(merged) == 8


def test_merge_artifacts_trims_existing_over_cap():
    """A pre-capped list is not assumed; a lowered cap still trims."""
    prior = [f"C:/data/artifacts/{index}.csv" for index in range(5)]
    merged = merge_artifacts(prior, [], max_artifacts=2)
    assert merged == [
        "C:/data/artifacts/3.csv",
        "C:/data/artifacts/4.csv",
    ]


def test_merge_artifacts_logs_drop_count_only(
    caplog: pytest.LogCaptureFixture,
):
    """Debug notes how many paths dropped, not which files."""
    caplog.set_level(logging.DEBUG, logger="src.agent.summarize")
    merge_artifacts(
        ["C:/data/artifacts/old.csv"],
        ["C:/data/artifacts/new.csv"],
        max_artifacts=1,
    )
    assert "dropped 1 older path(s)" in caplog.text
    assert "old.csv" not in caplog.text


def test_merge_artifacts_below_one_raises():
    """A non-positive cap is not silently treated as keep-all."""
    with pytest.raises(ValueError, match="at least 1"):
        merge_artifacts(["C:/data/artifacts/a.csv"], [], max_artifacts=0)


def test_stdout_without_artifacts_is_in_summary():
    """run_analysis_code still gets a summary when it only printed text."""
    stored = summarize_tool_result(
        "run_analysis_code",
        {"source_id": "file-abc", "stdout": "ok", "artifacts": []},
    )
    assert "stdout 2 chars" in stored["summary"]
    assert "ok" not in stored["summary"]


def test_unknown_result_keys_log_at_debug(caplog: pytest.LogCaptureFixture):
    """A renamed count field is logged; the summary stays the tool name."""
    caplog.set_level(logging.DEBUG, logger="src.agent.summarize")
    stored = summarize_tool_result("read_file_sample", {"n_rows": 3})
    assert stored["summary"] == "read_file_sample"
    assert "used no known keys" in caplog.text
    assert "n_rows" in caplog.text


def test_checkpoint_rejects_non_json_payload():
    """A DataFrame field fails closed instead of landing in state."""
    with pytest.raises(ValueError, match="JSON-safe"):
        checkpoint_tool_result(
            name="read_file_sample",
            result={"columns": pd.DataFrame({"a": [1]})},
            prior_artifacts=[],
        )
