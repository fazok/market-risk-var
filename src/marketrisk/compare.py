"""Comparing risk models with consistent scoring functions.

Passing a back-test only says a model is not rejected; it does not rank models.
Ranking needs a loss function for which the true risk measure is optimal:

* **Quantile (tick) loss** is consistent for VaR.
* **FZ0 loss** (Fissler & Ziegel 2016; Patton, Ziegel & Chen 2019) is consistent
  for the *pair* (VaR, ES). ES alone is not elicitable, so this joint loss is the
  right tool for comparing ES models.

The **Diebold-Mariano** test (with a Newey-West variance) then asks whether the
average loss difference between two models is statistically significant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def quantile_loss(returns, var, a: float) -> np.ndarray:
    """Tick loss for the return quantile q = -VaR at level tau = 1 - a."""
    y, q = np.asarray(returns, dtype=float), -np.asarray(var, dtype=float)
    tau = 1 - a
    return (tau - (y < q)) * (y - q)


def fz0_loss(returns, var, es, a: float) -> np.ndarray:
    """FZ0 joint loss for (VaR, ES) in the return convention (v = -VaR, e = -ES < 0)."""
    y = np.asarray(returns, dtype=float)
    v, e = -np.asarray(var, dtype=float), -np.asarray(es, dtype=float)
    tau = 1 - a
    hit = (y <= v).astype(float)
    return -hit * (v - y) / (tau * e) + v / e + np.log(-e) - 1


def newey_west_var(d: np.ndarray, lags: int | None = None) -> float:
    d = d - d.mean()
    n = len(d)
    lags = int(np.floor(4 * (n / 100) ** (2 / 9))) if lags is None else lags
    v = d @ d / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    return float(v)


def diebold_mariano(loss_1, loss_2, lags: int | None = None) -> dict[str, float]:
    """H0: equal expected loss. Negative statistic means model 1 is better."""
    d = np.asarray(loss_1, dtype=float) - np.asarray(loss_2, dtype=float)
    var = newey_west_var(d, lags)
    if var <= 0:
        return {"mean_diff": float(d.mean()), "dm": 0.0, "p_value": 1.0}
    dm = d.mean() / np.sqrt(var / len(d))
    return {"mean_diff": float(d.mean()), "dm": float(dm),
            "p_value": float(2 * stats.norm.sf(abs(dm)))}


def loss_table(res, var_level: float = 0.99, es_level: float = 0.975) -> pd.DataFrame:
    """Per-day losses for every model: quantile loss at ``var_level`` and FZ0 at ``es_level``."""
    out = {}
    for m in res.models:
        out[("quantile", m)] = quantile_loss(res.returns, res.var[var_level][m], var_level)
        out[("fz0", m)] = fz0_loss(res.returns, res.var[es_level][m], res.es[es_level][m],
                                   es_level)
    return pd.DataFrame(out, index=res.returns.index)


def ranking(losses: pd.DataFrame) -> pd.DataFrame:
    """Average losses, ranks, and a DM test of each model against the best (by FZ0)."""
    q = losses["quantile"].mean()
    fz = losses["fz0"].mean()
    best = fz.idxmin()
    rows = []
    for m in fz.index:
        dm = (diebold_mariano(losses["fz0"][m], losses["fz0"][best]) if m != best
              else {"dm": np.nan, "p_value": np.nan})
        rows.append({"model": m, "quantile_loss_x1e4": 1e4 * q[m], "fz0_loss": fz[m],
                     "dm_vs_best": dm["dm"], "dm_p_value": dm["p_value"]})
    out = pd.DataFrame(rows).set_index("model").sort_values("fz0_loss")
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def dm_matrix(losses: pd.DataFrame, kind: str = "fz0") -> pd.DataFrame:
    """Pairwise DM statistics (row vs column; negative = row model better)."""
    L = losses[kind]
    names = list(L.columns)
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    for i in names:
        for j in names:
            if i != j:
                mat.loc[i, j] = diebold_mariano(L[i], L[j])["dm"]
    return mat
