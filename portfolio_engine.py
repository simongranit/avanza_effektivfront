from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ------------------------------
# Grundstatistik
# ------------------------------

def compute_annual_stats(returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Tar månadsavkastningar och räknar fram:
      - Årsavkastning (geometrisk approx via 12 * mean)
      - Årskovarians
    """
    mean_monthly = returns.mean()
    cov_monthly = returns.cov()

    mean_annual = mean_monthly * 12
    cov_annual = cov_monthly * 12

    return mean_annual, cov_annual


def max_drawdown_from_returns(
    returns: pd.Series,
) -> Tuple[
    float,
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    pd.Series,
    Optional[int],
    Optional[int],
]:
    """
    Beräkna max drawdown (som positiv andel) utifrån avkastningsserie.

    Returnerar (max_drawdown, peak_date, trough_date, drawdown_series,
    dagar_peak_till_botten, dagar_peak_till_återhämtning).
    """

    r = returns.dropna()
    if r.empty:
        return 0.0, None, None, pd.Series(dtype=float), None, None

    cumulative = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1

    trough_date = drawdown.idxmin()
    peak_date = running_max.loc[:trough_date].idxmax()
    max_dd = float(abs(drawdown.min()))

    days_to_trough = None
    if peak_date is not None and trough_date is not None:
        if isinstance(r.index, pd.DatetimeIndex):
            days_to_trough = int((trough_date - peak_date).days)
        else:
            days_to_trough = int(r.index.get_loc(trough_date) - r.index.get_loc(peak_date))

    days_to_recovery = None
    if peak_date is not None and trough_date is not None:
        target_level = running_max.loc[peak_date]
        post_trough = cumulative.loc[trough_date:]
        recovery_candidates = post_trough[post_trough >= target_level]
        if not recovery_candidates.empty:
            recovery_date = recovery_candidates.index[0]
            if isinstance(r.index, pd.DatetimeIndex):
                days_to_recovery = int((recovery_date - peak_date).days)
            else:
                days_to_recovery = int(
                    r.index.get_loc(recovery_date) - r.index.get_loc(peak_date)
                )

    return (
        max_dd,
        peak_date,
        trough_date,
        drawdown,
        days_to_trough,
        days_to_recovery,
    )


def max_drawdown_from_prices(
    prices: pd.Series,
) -> Tuple[
    float,
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    pd.Series,
    Optional[int],
    Optional[int],
]:
    """Max drawdown baserat på prisserie."""

    p = prices.sort_index().dropna()
    if p.empty:
        return 0.0, None, None, pd.Series(dtype=float), None, None

    returns = p.pct_change().dropna()
    return max_drawdown_from_returns(returns)


def portfolio_max_drawdown(
    returns: pd.DataFrame, weights: np.ndarray
) -> Tuple[
    float,
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    pd.Series,
    Optional[int],
    Optional[int],
]:
    """Max drawdown för en portfölj givet historiska avkastningar och vikter."""

    port_returns = returns.mul(weights, axis=1).sum(axis=1)
    return max_drawdown_from_returns(port_returns)


def bootstrap_max_drawdown(
    portfolio_returns: pd.Series, n_simulations: int = 300
) -> float:
    """
    Grov uppskattning av simulerad max drawdown via bootstrap på historiska
    portföljavkastningar. Returnerar medianen av simulerade max drawdowns.
    """

    r = portfolio_returns.dropna().values
    if r.size == 0:
        return float("nan")

    draws = []
    n = len(r)
    for _ in range(n_simulations):
        sample = np.random.choice(r, size=n, replace=True)
        dd, _, _, _, _, _ = max_drawdown_from_returns(pd.Series(sample))
        draws.append(dd)

    if not draws:
        return float("nan")

    return float(np.median(draws))


# ------------------------------
# Monte Carlo-portföljer
# ------------------------------

def sample_weights_with_constraints(
    n_assets: int,
    swe_eq_idx: np.ndarray,
    for_eq_idx: np.ndarray,
    bond_idx: np.ndarray,
    max_weight: float,
    max_equity: float,
    max_foreign_equity_frac: float,
    min_equity: float = 0.0,
    max_tries: int = 3000,
) -> np.ndarray:
    """
    Slumpar vikter med:
      - sum(w) = 1
      - 0 <= w_i <= max_weight
      - min_equity <= sum(equity) <= max_equity
      - (sum(foreign_equity) / sum(equity)) <= max_foreign_equity_frac
    """
    n_se = len(swe_eq_idx)
    n_fe = len(for_eq_idx)
    n_b = len(bond_idx)

    for _ in range(max_tries):
        w = np.zeros(n_assets)

        # 1) Slumpa total aktieandel
        if (n_se + n_fe) > 0:
            equity_share = np.random.uniform(min_equity, max_equity)
        else:
            equity_share = 0.0
        bond_share = 1.0 - equity_share

        # Grovkoll per fond-cap
        if (n_se + n_fe) > 0 and (n_se + n_fe) * max_weight < equity_share - 1e-9:
            continue
        if n_b > 0 and n_b * max_weight < bond_share - 1e-9:
            continue
        if n_b == 0 and bond_share > 1e-9:
            # Kräver ränta men har inga räntefonder
            continue

        # 2) Dela upp aktiedelen i svensk & utländsk med constraint för utländsk del
        if equity_share > 0 and n_fe > 0 and max_foreign_equity_frac > 0:
            u = np.random.uniform(0.0, max_foreign_equity_frac)
            foreign_share = equity_share * u
        else:
            foreign_share = 0.0

        swe_share = equity_share - foreign_share

        # Feasibility per fond inom grupper
        if n_fe > 0 and n_fe * max_weight < foreign_share - 1e-9:
            continue
        if n_se > 0 and n_se * max_weight < swe_share - 1e-9:
            continue
        if n_se == 0 and swe_share > 1e-9:
            continue

        # 3) Slumpa inom svensk aktiedel
        if n_se > 0 and swe_share > 0:
            we_s = np.random.dirichlet(np.ones(n_se)) * swe_share
            for _ in range(20):
                over = we_s > max_weight + 1e-9
                if not over.any():
                    break
                excess = we_s[over] - max_weight
                we_s[over] = max_weight
                under = ~over
                if under.any():
                    we_s[under] += excess.sum() * (we_s[under] / we_s[under].sum())
                else:
                    break
        else:
            we_s = np.zeros(n_se)

        # 4) Slumpa inom utländsk aktiedel
        if n_fe > 0 and foreign_share > 0:
            we_f = np.random.dirichlet(np.ones(n_fe)) * foreign_share
            for _ in range(20):
                over = we_f > max_weight + 1e-9
                if not over.any():
                    break
                excess = we_f[over] - max_weight
                we_f[over] = max_weight
                under = ~over
                if under.any():
                    we_f[under] += excess.sum() * (we_f[under] / we_f[under].sum())
                else:
                    break
        else:
            we_f = np.zeros(n_fe)

        # 5) Slumpa inom räntedel
        if n_b > 0 and bond_share > 0:
            wb = np.random.dirichlet(np.ones(n_b)) * bond_share
            for _ in range(20):
                over = wb > max_weight + 1e-9
                if not over.any():
                    break
                excess = wb[over] - max_weight
                wb[over] = max_weight
                under = ~over
                if under.any():
                    wb[under] += excess.sum() * (wb[under] / wb[under].sum())
                else:
                    break
        else:
            wb = np.zeros(n_b)

        w[swe_eq_idx] = we_s
        w[for_eq_idx] = we_f
        w[bond_idx] = wb

        # Slutkoller
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            continue
        if (w > max_weight + 1e-6).any():
            continue

        total_equity = w[swe_eq_idx].sum() + w[for_eq_idx].sum()
        foreign_equity = w[for_eq_idx].sum()

        if total_equity > 0:
            frac_foreign_equity = foreign_equity / total_equity
            if frac_foreign_equity > max_foreign_equity_frac + 1e-6:
                continue

        if not (min_equity - 1e-6 <= total_equity <= max_equity + 1e-6):
            continue

        return w

    raise RuntimeError("Kunde inte generera vikter med givna restriktioner (inkl. utländsk aktiedel).")


def simulate_portfolios(
    returns: pd.DataFrame,
    swe_eq_mask: np.ndarray,
    for_eq_mask: np.ndarray,
    max_weight: float,
    max_equity_share: float,
    max_foreign_equity_frac: float,
    n_portfolios: int,
    rf: float,
    min_equity_share: float = 0.0,
    max_drawdown: float | None = None,
) -> Dict[str, np.ndarray]:
    """
    Monte Carlo-simulering av portföljer.
    """
    mean_returns, cov_matrix = compute_annual_stats(returns)
    n_assets = returns.shape[1]

    swe_idx = np.where(swe_eq_mask)[0]
    for_idx = np.where(for_eq_mask)[0]
    bond_idx = np.where(~(swe_eq_mask | for_eq_mask))[0]

    port_returns, port_vols, sharpes, weights, drawdowns = [], [], [], [], []

    attempts = 0
    max_attempts = max(n_portfolios * 10, 1000)
    while len(port_returns) < n_portfolios and attempts < max_attempts:
        attempts += 1
        w = sample_weights_with_constraints(
            n_assets=n_assets,
            swe_eq_idx=swe_idx,
            for_eq_idx=for_idx,
            bond_idx=bond_idx,
            max_weight=max_weight,
            max_equity=max_equity_share,
            max_foreign_equity_frac=max_foreign_equity_frac,
            min_equity=min_equity_share,
        )

        dd, *_ = portfolio_max_drawdown(returns, w)
        if max_drawdown is not None and dd > max_drawdown:
            continue

        r = float(np.dot(w, mean_returns.values))
        v = float(np.sqrt(w @ cov_matrix.values @ w))
        s = (r - rf) / v if v > 0 else float("nan")

        port_returns.append(r)
        port_vols.append(v)
        sharpes.append(s)
        weights.append(w)
        drawdowns.append(dd)

    if len(port_returns) < n_portfolios:
        raise RuntimeError(
            "Kunde inte hitta tillräckligt många portföljer som uppfyller drawdown-kravet."
        )

    return {
        "returns": np.array(port_returns),
        "vols": np.array(port_vols),
        "sharpes": np.array(sharpes),
        "weights": np.array(weights),
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix,
        "drawdowns": np.array(drawdowns),
    }


# ------------------------------
# Teoretisk optimering (SLSQP)
# ------------------------------

def _build_constraints(
    mean_returns: pd.Series,
    swe_eq_mask: np.ndarray,
    for_eq_mask: np.ndarray,
    max_equity_share: float,
    max_foreign_equity_frac: float,
    target_return: float | None = None,
):
    """
    Bygger constraints-listan för scipy.optimize.minimize.
    """
    n = len(mean_returns)
    eq_idx = np.where(swe_eq_mask | for_eq_mask)[0]
    fe_idx = np.where(for_eq_mask)[0]

    cons = []

    # Summa vikter = 1
    cons.append(
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    )

    # Max andel aktier
    if len(eq_idx) > 0:
        def equity_constraint(w, idx=eq_idx, max_eq=max_equity_share):
            return max_eq - np.sum(w[idx])  # >= 0
        cons.append({"type": "ineq", "fun": equity_constraint})

    # Max andel utländska aktier av aktiedelen
    if len(eq_idx) > 0 and len(fe_idx) > 0 and max_foreign_equity_frac < 1.0:
        def foreign_constraint(w, e_idx=eq_idx, f_idx=fe_idx, mfe=max_foreign_equity_frac):
            se = np.sum(w[e_idx])
            if se < 1e-8:
                return 0.0  # inga aktier => constraint ok
            foreign = np.sum(w[f_idx])
            return mfe * se - foreign  # >= 0

        cons.append({"type": "ineq", "fun": foreign_constraint})

    # Targetreturn om vi bygger front
    if target_return is not None:
        def target_ret_constraint(w, mu=mean_returns.values, tr=target_return):
            return float(np.dot(w, mu) - tr)
        cons.append({"type": "eq", "fun": target_ret_constraint})

    return cons


def min_variance_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    swe_eq_mask: np.ndarray,
    for_eq_mask: np.ndarray,
    max_equity_share: float,
    max_foreign_equity_frac: float,
    max_weight: float,
) -> np.ndarray:
    """
    Minsta-varians-portfölj givet constraints.
    """
    n = len(mean_returns)
    bounds = [(0.0, max_weight)] * n
    w0 = np.repeat(1.0 / n, n)

    cons = _build_constraints(
        mean_returns=mean_returns,
        swe_eq_mask=swe_eq_mask,
        for_eq_mask=for_eq_mask,
        max_equity_share=max_equity_share,
        max_foreign_equity_frac=max_foreign_equity_frac,
        target_return=None,
    )

    def obj(w, cov=cov_matrix.values):
        return float(w @ cov @ w)

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Optimering misslyckades (min varians): {res.message}")
    return res.x


def max_sharpe_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    swe_eq_mask: np.ndarray,
    for_eq_mask: np.ndarray,
    max_equity_share: float,
    max_foreign_equity_frac: float,
    max_weight: float,
    rf: float,
) -> np.ndarray:
    """
    Max Sharpe-portfölj givet constraints.
    """
    n = len(mean_returns)
    bounds = [(0.0, max_weight)] * n
    w0 = np.repeat(1.0 / n, n)

    cons = _build_constraints(
        mean_returns=mean_returns,
        swe_eq_mask=swe_eq_mask,
        for_eq_mask=for_eq_mask,
        max_equity_share=max_equity_share,
        max_foreign_equity_frac=max_foreign_equity_frac,
        target_return=None,
    )

    mu = mean_returns.values
    cov = cov_matrix.values

    def neg_sharpe(w, mu=mu, cov=cov, rf=rf):
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        if v <= 0:
            return 1e6
        return - (r - rf) / v

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Optimering misslyckades (max Sharpe): {res.message}")
    return res.x


def efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    swe_eq_mask: np.ndarray,
    for_eq_mask: np.ndarray,
    max_equity_share: float,
    max_foreign_equity_frac: float,
    max_weight: float,
    n_points: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximerar den effektiva fronten genom att minimera variansen
    för ett antal olika target returns.
    Returnerar (vol_array, ret_array).
    """
    mu = mean_returns.values
    cov = cov_matrix.values
    n = len(mean_returns)

    # Ungefärliga min/max returns (baserat på unconstrained portföljer)
    w_eq = np.repeat(1.0 / n, n)
    r_mid = float(w_eq @ mu)
    r_min = min(mu.min(), r_mid * 0.5)
    r_max = mu.max()

    target_rets = np.linspace(r_min, r_max, n_points)
    bounds = [(0.0, max_weight)] * n

    vols, rets = [], []

    for tr in target_rets:
        w0 = w_eq.copy()
        cons = _build_constraints(
            mean_returns=mean_returns,
            swe_eq_mask=swe_eq_mask,
            for_eq_mask=for_eq_mask,
            max_equity_share=max_equity_share,
            max_foreign_equity_frac=max_foreign_equity_frac,
            target_return=tr,
        )

        def obj(w, cov=cov):
            return float(w @ cov @ w)

        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success:
            # hoppa över infeasible punkter
            continue

        w = res.x
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        vols.append(v)
        rets.append(r)

    return np.array(vols), np.array(rets)


def describe_portfolio(
    weights: np.ndarray,
    asset_names: list[str],
    annual_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Skapar en Serie med avkastning, risk och vikter per fond.
    """
    w = weights
    mu = annual_returns.values
    cov = cov_matrix.values

    r = float(w @ mu)
    v = float(np.sqrt(w @ cov @ w))

    s = pd.Series({"Förväntad årsavkastning": r, "Årsvolatilitet": v})
    for i, name in enumerate(asset_names):
        s[f"Vikt {name}"] = w[i]
    return s
