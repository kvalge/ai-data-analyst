# test_profile_pause.py

"""Tests for Guided profile interrupt display helpers. No Streamlit run."""

from src.agent.profile_steps import SECTION_SCHEMA
from src.ui.profile_pause import profile_result_for_pause


def test_pause_uses_checkpointed_summary_when_present():
    """Later pauses prefer the stored compact profile over the section stub."""
    stored = {
        "cached": True,
        "sample_row_count": 3,
        "schema": {"columns": ["date"], "dtypes": {}, "null_counts": {}},
        "dq": {"duplicates": {"duplicate_row_count": 0}},
        "eda": {"summary": {}},
    }
    result = profile_result_for_pause(
        {"step": "dq", "section": {"duplicates": {"duplicate_row_count": 1}}},
        stored,
    )
    assert result is stored


def test_first_schema_pause_uses_interrupt_section():
    """The first pause has no checkpointed summary yet."""
    payload = {
        "step": SECTION_SCHEMA,
        "cached": False,
        "sample_row_count": 3,
        "section": {
            "columns": ["date", "region", "revenue"],
            "dtypes": {},
            "null_counts": {},
        },
    }
    result = profile_result_for_pause(payload, None)
    assert result is not None
    assert result["schema"]["columns"] == ["date", "region", "revenue"]
    assert "rows" not in result
