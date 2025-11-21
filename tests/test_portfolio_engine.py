import numpy as np
import pandas as pd

from portfolio_engine import (
    bootstrap_max_drawdown,
    compute_annual_stats,
    max_drawdown_from_returns,
    portfolio_max_drawdown,
)


def test_compute_annual_stats_scales_returns_and_covariance():
    data = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03],
            "B": [0.00, -0.01, 0.02],
        }
    )

    mean_annual, cov_annual = compute_annual_stats(data)

    pd.testing.assert_series_equal(mean_annual, data.mean() * 12)
    pd.testing.assert_frame_equal(cov_annual, data.cov() * 12)


def test_max_drawdown_from_returns_handles_empty_series():
    max_dd, peak, trough, dd_series = max_drawdown_from_returns(pd.Series(dtype=float))

    assert max_dd == 0.0
    assert peak is None
    assert trough is None
    assert dd_series.empty


def test_portfolio_max_drawdown_combines_weights():
    returns = pd.DataFrame(
        {
            "A": [0.1, -0.05, 0.02],
            "B": [0.0, 0.01, -0.01],
        }
    )
    weights = np.array([0.6, 0.4])

    max_dd, _, _, _ = portfolio_max_drawdown(returns, weights)

    expected_portfolio = returns.mul(weights, axis=1).sum(axis=1)
    expected_max_dd, _, _, _ = max_drawdown_from_returns(expected_portfolio)

    assert np.isclose(max_dd, expected_max_dd)


def test_bootstrap_max_drawdown_returns_median():
    np.random.seed(42)
    portfolio_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])

    max_dd = bootstrap_max_drawdown(portfolio_returns, n_simulations=100)

    assert 0 <= max_dd <= 1
