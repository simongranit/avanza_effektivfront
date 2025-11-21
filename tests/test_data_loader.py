import pandas as pd

from data_loader import build_return_matrix


def test_build_return_matrix_resamples_month_end():
    dates = pd.date_range("2024-01-01", periods=4, freq="15D")
    series_a = pd.Series([100, 105, 102, 110], index=dates, name="Fund A")
    series_b = pd.Series([200, 195, 210, 205], index=dates, name="Fund B")

    returns = build_return_matrix({"Fund A": series_a, "Fund B": series_b}, freq="ME")

    assert list(returns.columns) == ["Fund A", "Fund B"]
    assert not returns.isnull().any().any()
    # Should have pct_change over at least two resampled points
    assert len(returns) >= 1
