"""Sanity checks for committed test fixtures."""

import pandas as pd


def test_sample_sales_csv_loads(sample_sales_csv):
    """The sample sales fixture is a readable CSV with the expected columns."""
    frame = pd.read_csv(sample_sales_csv)

    assert list(frame.columns) == ["date", "region", "revenue"]
    assert len(frame) == 5
    assert frame["revenue"].dtype.kind in "iu"
