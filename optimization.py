# optimization.py
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from portfolio_engine import portfolio_max_drawdown


def portfolio_stats(weights: np.ndarray,
                    mean_returns: pd.Series,
                    cov_matrix: pd.DataFrame) -> tuple[float, float]:
    """Returnera (förväntad årsavkastning, årsvolatilitet) för givna vikter."""
    w = np.asarray(weights).reshape(-1)
    mu = mean_returns.values
    Sigma = cov_matrix.values

    r = float(w @ mu)
    v = float(np.sqrt(w @ Sigma @ w))
    return r, v


# ---------------- UNCONSTRAINED FRONT ---------------- #

def efficient_frontier_unconstrained(mean_returns: pd.Series,
                                     cov_matrix: pd.DataFrame,
                                     rf: float,
                                     n_points: int = 60):
    """
    Effektiv front utan dina praktiska constraints (bara:
      - w_i >= 0
      - sum w_i = 1
    ).

    Returnerar:
      frontier_risk, frontier_return, w_min_var, w_max_sharpe
    """
    mu = mean_returns.values
    Sigma = cov_matrix.values
    n = len(mu)

    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, Sigma)

    base_constraints = [cp.sum(w) == 1, w >= 0]

    # Välj målavkastningar i intervallet [min(mu), max(mu)]
    target_returns = np.linspace(mu.min(), mu.max(), n_points)

    frontier_ret = []
    frontier_risk = []
    frontier_weights = []

    for R in target_returns:
        prob = cp.Problem(cp.Minimize(risk),
                          base_constraints + [ret >= R])
        prob.solve(solver=cp.ECOS, verbose=False)
        if w.value is None:
            continue
        w_opt = np.array(w.value).reshape(-1)
        r, v = portfolio_stats(w_opt, mean_returns, cov_matrix)
        frontier_ret.append(r)
        frontier_risk.append(v)
        frontier_weights.append(w_opt)

    frontier_ret = np.array(frontier_ret)
    frontier_risk = np.array(frontier_risk)
    frontier_weights = np.vstack(frontier_weights)

    # Minsta varians = lägsta risk på fronten
    min_idx = np.argmin(frontier_risk)
    w_min_var = frontier_weights[min_idx]

    # Max Sharpe = bästa sharpe längs fronten
    excess_ret = frontier_ret - rf
    sharpe = excess_ret / frontier_risk
    max_idx = np.nanargmax(sharpe)
    w_max_sharpe = frontier_weights[max_idx]

    return frontier_risk, frontier_ret, w_min_var, w_max_sharpe


# ---------------- CONSTRAINED FRONT ---------------- #

def efficient_frontier_constrained(mean_returns: pd.Series,
                                   cov_matrix: pd.DataFrame,
                                   rf: float,
                                   swe_eq_mask: np.ndarray,
                                   for_eq_mask: np.ndarray,
                                   max_weight: float,
                                   max_equity_share: float,
                                   max_foreign_equity_frac: float,
                                   n_points: int = 40,
                                   returns_history: pd.DataFrame | None = None,
                                   max_drawdown: Optional[float] = None):
    """
    Effektiv front med SAMMA constraints som Monte Carlo:

      - w_i >= 0
      - w_i <= max_weight
      - sum w_i = 1
      - equity_share = sum(svensk + utländsk aktie) <= max_equity_share
      - foreign_share <= max_foreign_equity_frac * equity_share

    Returnerar:
      frontier_risk, frontier_return, w_min_var, w_max_sharpe
    """
    mu = mean_returns.values
    Sigma = cov_matrix.values
    n = len(mu)

    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, Sigma)

    swe_idx = np.where(swe_eq_mask)[0]
    for_idx = np.where(for_eq_mask)[0]
    eq_idx = np.where(swe_eq_mask | for_eq_mask)[0]

    # Bas-constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
    ]

    if len(eq_idx) > 0:
        equity_share = cp.sum(w[eq_idx])
        constraints.append(equity_share <= max_equity_share)

        if len(for_idx) > 0 and max_foreign_equity_frac < 1.0:
            foreign_share = cp.sum(w[for_idx])
            # foreign_share <= max_foreign_equity_frac * equity_share
            # <=> (1 - max_fe)*foreign_share - max_fe*swedish_share <= 0
            # Men enklast att skriva direkt:
            constraints.append(foreign_share <= max_foreign_equity_frac * equity_share)

    # Välj målavkastningar INOM det som verkar rimligt
    target_returns = np.linspace(mu.min(), mu.max(), n_points)

    frontier_ret = []
    frontier_risk = []
    frontier_weights = []

    for R in target_returns:
        prob = cp.Problem(cp.Minimize(risk),
                          constraints + [ret >= R])
        prob.solve(solver=cp.ECOS, verbose=False)

        if w.value is None:
            continue

        w_opt = np.array(w.value).reshape(-1)

        if max_drawdown is not None and returns_history is not None:
            dd, _, _, _ = portfolio_max_drawdown(returns_history, w_opt)
            if dd > max_drawdown:
                continue

        r, v = portfolio_stats(w_opt, mean_returns, cov_matrix)
        frontier_ret.append(r)
        frontier_risk.append(v)
        frontier_weights.append(w_opt)

    if not frontier_weights:
        # Inga lösningar hittades – t.ex. för hårda constraints
        return (np.array([]), np.array([]),
                np.full(n, np.nan), np.full(n, np.nan))

    frontier_ret = np.array(frontier_ret)
    frontier_risk = np.array(frontier_risk)
    frontier_weights = np.vstack(frontier_weights)

    # Minsta varians längs constrained front
    min_idx = np.argmin(frontier_risk)
    w_min_var = frontier_weights[min_idx]

    # Max Sharpe längs constrained front
    excess_ret = frontier_ret - rf
    sharpe = excess_ret / frontier_risk
    max_idx = np.nanargmax(sharpe)
    w_max_sharpe = frontier_weights[max_idx]

    return frontier_risk, frontier_ret, w_min_var, w_max_sharpe
