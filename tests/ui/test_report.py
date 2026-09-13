# test_report.py

"""Tests for the last-summary markdown report. No row dumps."""

from __future__ import annotations

from datetime import datetime, timezone

from src.ui.report import (
    build_report_markdown,
    last_tool_summary,
    report_download_name,
)


def test_last_tool_summary_reads_summary_field():
    """Only the checkpointed summary line is used."""
    assert (
        last_tool_summary({"summary": "run_analysis_code: 1 artifact(s)"})
        == "run_analysis_code: 1 artifact(s)"
    )


def test_last_tool_summary_ignores_other_keys():
    """Snippet or row-like keys are not copied into the report."""
    text = last_tool_summary(
        {
            "summary": "retrieve_domain_context: 1 chunk(s)",
            "chunks": [{"text": "secret glossary line"}],
            "rows": [{"date": "2024-01-01", "region": "north", "revenue": 10}],
        }
    )
    assert text == "retrieve_domain_context: 1 chunk(s)"
    assert "secret" not in text
    assert "north" not in text


def test_build_report_markdown_includes_summary_and_paths():
    """The download lists the last summary and artifact paths."""
    text = build_report_markdown(
        summary="run_analysis_code: 1 artifact(s)",
        artifacts=[r"C:\data\artifacts\summary.csv"],
    )
    assert "# Analysis report" in text
    assert "run_analysis_code: 1 artifact(s)" in text
    assert r"C:\data\artifacts\summary.csv" in text
    assert "date,region,revenue" not in text


def test_build_report_markdown_is_empty_without_content():
    """No button payload when there is nothing to export."""
    assert build_report_markdown(summary="  ", artifacts=["", "  "]) == ""


def test_report_download_name_includes_thread_and_stamp():
    """Downloads from different threads or times get different filenames."""
    when = datetime(2026, 9, 14, 0, 42, tzinfo=timezone.utc)
    name = report_download_name(
        "93d18163-3337-4463-b989-1dc9c41ef684", when=when
    )
    assert name == "report_93d18163_20260914T004200Z.md"


def test_report_download_name_uses_fallback_prefix():
    """A blank thread id still produces a usable filename."""
    when = datetime(2026, 9, 14, 0, 42, tzinfo=timezone.utc)
    assert report_download_name("   ", when=when) == (
        "report_thread_20260914T004200Z.md"
    )


def test_build_report_markdown_omits_empty_summary_section():
    """Artifact-only state still produces a report."""
    text = build_report_markdown(
        summary="",
        artifacts=[r"C:\data\artifacts\chart.png"],
    )
    assert "## Last summary" not in text
    assert "chart.png" in text
