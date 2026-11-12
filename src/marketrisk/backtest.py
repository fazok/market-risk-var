"""Rolling out-of-sample forecasts and regulatory back-tests.

* Kupiec (1995) proportion-of-failures test: is the exception rate right?
* Christoffersen (1998) independence and conditional-coverage tests: do
  exceptions cluster?
* Basel traffic light: exceptions over the last 250 days at 99%
  (green 0-4, yellow 5-9, red 10+), with the capital multiplier plus-factor.
* Acerbi & Szekely (2014) Z2 test of Expected Shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .models import VaRModel


@dataclass
class BacktestResult:
    returns: pd.Series  # realised next-day portfolio returns
    var: dict[float, pd.DataFrame] = field(default_factory=dict)  # level -> dates x models
    es: dict[float, pd.DataFrame] = field(default_factory=dict)

    @property
    def models(self) -> list[str]:
        return list(next(iter(self.var.values())).columns)

    @property
    def losses(self) -> pd.Series:
        return -self.returns

    def exceptions(self, level: float) -> pd.DataFrame:
        v = self.var[level]
        return v.lt(self.losses, axis=0).astype(int)


def run_rolling(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    models: list[VaRModel],
    window: int = 500,
    levels: tuple[float, ...] = (0.99, 0.975),
    progress: bool = False,
) -> BacktestResult:
    """Forecast day t from the ``window`` days before it, for every t after the window."""
    R = asset_returns.to_numpy()
    w = np.asarray(weights, dtype=float)
    dates = asset_returns.index[window:]
    var = {a: np.empty((len(dates), len(models))) for a in levels}
    es = {a: np.empty((len(dates), len(models))) for a in levels}
    for i, t in enumerate(range(window, len(R))):
        R_win = R[t - window:t]
        for j, m in enumerate(models):
            fc = m.forecast(R_win, w, levels)
            for a in levels:
                var[a][i, j], es[a][i, j] = fc[a]
        if progress and i % 500 == 0:
            print(f"  {i:>5}/{len(dates)}  {dates[i].date()}")
    names = [m.name for m in models]
    return BacktestResult(
        returns=pd.Series(R[window:] @ w, index=dates, name="return"),
        var={a: pd.DataFrame(var[a], index=dates, columns=names) for a in levels},
        es={a: pd.DataFrame(es[a], index=dates, columns=names) for a in levels},
    )


# --- Coverage tests ----------------------------------------------------------------


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(n: int, x: int, p: float) -> dict[str, float]:
    """Unconditional coverage: LR = -2 ln[L(p) / L(x/n)] ~ chi2(1)."""
    pi = x / n
    ll_null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll_alt = _xlogy(n - x, 1 - pi) + _xlogy(x, pi)
    lr = -2 * (ll_null - ll_alt)
    return {"n": n, "exceptions": x, "expected": n * p, "rate": pi,
            "lr": lr, "p_value": float(stats.chi2.sf(lr, 1))}


def christoffersen(hits: np.ndarray, p: float) -> dict[str, float]:
    """Independence (first-order Markov) and conditional coverage tests."""
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_ind = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
              + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    lr_ind = max(-2 * (ll_null - ll_ind), 0.0)
    lr_pof = kupiec_pof(len(h), int(h.sum()), p)["lr"]
    lr_cc = lr_pof + lr_ind
    return {"lr_ind": lr_ind, "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "lr_cc": lr_cc, "p_cc": float(stats.chi2.sf(lr_cc, 2)),
            "p_exception_after_exception": pi11}


# --- Basel traffic light -----------------------------------------------------------

PLUS_FACTOR = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def traffic_light_zone(exceptions_250: int) -> str:
    if exceptions_250 <= 4:
        return "green"
    if exceptions_250 <= 9:
        return "yellow"
    return "red"


def plus_factor(exceptions_250: int) -> float:
    if exceptions_250 <= 4:
        return 0.0
    return PLUS_FACTOR.get(exceptions_250, 1.0)


def rolling_exceptions(hits: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    return hits.rolling(window).sum().dropna()


def zone_shares(hits: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    """Share of days each model spends in each traffic-light zone."""
    counts = rolling_exceptions(hits, window)
    zones = counts.apply(lambda c: c.astype(int).map(traffic_light_zone))
    out = pd.DataFrame({z: (zones == z).mean() for z in ["green", "yellow", "red"]})
    out["max_exceptions_250d"] = counts.max().astype(int)
    return out


# --- Expected Shortfall ------------------------------------------------------------

# Acerbi & Szekely (2014) report that Z2 critical values are stable across
# distributions: about 0.70 at 5% and 1.80 at 0.01% (sign flipped to our loss convention).
Z2_YELLOW, Z2_RED = 0.70, 1.80


def acerbi_szekely_z2(losses, var, es, a: float) -> float:
    """Z2 = sum(L_t 1{L_t > VaR_t} / ES_t) / (T (1 - a)) - 1.

    Zero in expectation when ES is correct; positive when ES underestimates risk.
    """
    L, v, e = (np.asarray(x, dtype=float) for x in (losses, var, es))
    return float(np.sum(L * (L > v) / e) / (len(L) * (1 - a)) - 1)


def z2_zone(z2: float) -> str:
    if z2 < Z2_YELLOW:
        return "green"
    return "yellow" if z2 < Z2_RED else "red"


# --- Summary -----------------------------------------------------------------------


def summary_table(res: BacktestResult, var_level: float = 0.99,
                  es_level: float = 0.975) -> pd.DataFrame:
    """One row per model: coverage, clustering, traffic light and ES tests."""
    p = 1 - var_level
    hits = res.exceptions(var_level)
    zones = zone_shares(hits)
    rows = []
    for m in res.models:
        h = hits[m].to_numpy()
        k = kupiec_pof(len(h), int(h.sum()), p)
        c = christoffersen(h, p)
        z2 = acerbi_szekely_z2(res.losses, res.var[es_level][m], res.es[es_level][m], es_level)
        rows.append({
            "model": m,
            "exceptions": k["exceptions"],
            "expected": round(k["expected"], 1),
            "rate_%": 100 * k["rate"],
            "kupiec_p": k["p_value"],
            "christ_ind_p": c["p_ind"],
            "christ_cc_p": c["p_cc"],
            "days_red_%": 100 * zones.loc[m, "red"],
            "max_exc_250d": int(zones.loc[m, "max_exceptions_250d"]),
            "es_z2": z2,
            "es_zone": z2_zone(z2),
            "avg_var_%": 100 * res.var[var_level][m].mean(),
        })
    return pd.DataFrame(rows).set_index("model")


def period_exceptions(res: BacktestResult, periods: dict[str, tuple[str, str]],
                      level: float = 0.99) -> pd.DataFrame:
    """Exception rate (%) by stress period, to see which models break in crises."""
    hits = res.exceptions(level)
    out = {}
    for name, (start, end) in periods.items():
        part = hits.loc[start:end]
        if len(part):
            out[f"{name} ({len(part)}d)"] = 100 * part.mean()
    df = pd.DataFrame(out).rename_axis("model")
    df[f"full ({len(hits)}d)"] = 100 * hits.mean()
    return df


STRESS_PERIODS = {
    "GFC 2008": ("2008-09-01", "2009-03-31"),
    "Euro crisis 2011": ("2011-07-01", "2011-12-31"),
    "COVID 2020": ("2020-02-15", "2020-05-31"),
    "Rates 2022": ("2022-01-01", "2022-12-31"),
}


def ten_day_scaling(res: BacktestResult, level: float = 0.99, horizon: int = 10) -> pd.DataFrame:
    """Back-test the square-root-of-time rule.

    Each model's 1-day VaR x sqrt(10), issued at day t, is compared with the realised
    compounded portfolio loss over days t..t+9, on non-overlapping blocks.
    """
    r = res.returns
    growth = np.log1p(r).rolling(horizon).sum().shift(-(horizon - 1))
    loss_h = -(np.expm1(growth))
    loss_h = loss_h.iloc[::horizon].dropna()
    scaled = res.var[level].loc[loss_h.index] * np.sqrt(horizon)
    hits = scaled.lt(loss_h, axis=0)
    p = 1 - level
    rows = []
    for m in res.models:
        k = kupiec_pof(len(hits), int(hits[m].sum()), p)
        rows.append({"model": m, "blocks": k["n"], "exceptions": k["exceptions"],
                     "rate_%": 100 * k["rate"], "kupiec_p": k["p_value"]})
    return pd.DataFrame(rows).set_index("model")
