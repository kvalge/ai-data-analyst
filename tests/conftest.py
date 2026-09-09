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


@pytest.fixture
def sample_nulls_csv(fixtures_dir: Path) -> Path:
    """Path to the tiny CSV used for null-count DQ tests."""
    return fixtures_dir / "sample_nulls.csv"


@pytest.fixture
def sample_duplicates_csv(fixtures_dir: Path) -> Path:
    """Path to the tiny CSV used for full-row duplicate DQ tests."""
    return fixtures_dir / "sample_duplicates.csv"


@pytest.fixture
def sample_type_mismatches_csv(fixtures_dir: Path) -> Path:
    """Path to the tiny sales-shaped CSV used for type-mismatch DQ tests."""
    return fixtures_dir / "sample_type_mismatches.csv"


@pytest.fixture
def sample_inconsistent_formatting_csv(fixtures_dir: Path) -> Path:
    """Path to the tiny sales-shaped CSV used for formatting DQ tests."""
    return fixtures_dir / "sample_inconsistent_formatting.csv"


@pytest.fixture
def sample_outliers_csv(fixtures_dir: Path) -> Path:
    """Path to the tiny sales-shaped CSV used for outlier DQ tests."""
    return fixtures_dir / "sample_outliers.csv"
