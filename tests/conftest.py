# conftest.py

"""Shared pytest fixtures."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Directory containing committed test data files."""
    return FIXTURES_DIR


@pytest.fixture
def sample_sales_csv(fixtures_dir: Path) -> Path:
    """Path to the tiny sales CSV used across early tests."""
    return fixtures_dir / "sample_sales.csv"


@pytest.fixture
def sample_glossary_md(fixtures_dir: Path) -> Path:
    """Path to the tiny glossary markdown used for context-file tests."""
    return fixtures_dir / "sample_glossary.md"
