"""Seven one-day VaR / Expected Shortfall models.

Conventions: the loss is ``L = -r`` for a portfolio return ``r``. VaR and ES are
reported as **positive fractions of portfolio value**, so a 1% loss is 0.01.

* ``VaR_a = q_a(L)``: the loss exceeded with probability ``1 - a``.
* ``ES_a = E[L | L >= VaR_a]``: the average loss beyond VaR (the FRTB measure,
  at a = 97.5%).

Each model implements ``forecast(R, w, levels)``. ``R`` is the T x N window of
asset returns, ``w`` the weights, and the result is ``{level: (VaR, ES)}`` for
the next day. Univariate models work on the portfolio return series ``R @ w``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.signal import lfilter

Forecast = dict  # {level: (var, es)}


class VaRModel:
    name = "base"
    multivariate = False

    def forecast(self, R: np.ndarray, w: np.ndarray, levels) -> Forecast:
        return self.forecast_univariate(np.asarray(R) @ np.asarray(w), levels)

    def forecast_univariate(self, r: np.ndarray, levels) -> Forecast:  # pragma: no cover
        raise NotImplementedError


# --- Closed-form helpers ---------------------------------------------------------


def normal_var_es(mu: float, sigma: float, a: float) -> tuple[float, float]:
    z = stats.norm.ppf(a)
    return -mu + sigma * z, -mu + sigma * stats.norm.pdf(z) / (1 - a)


def std_t_var_es(nu: float, a: float) -> tuple[float, float]:
    """VaR and ES of a Student-t rescaled to unit variance."""
    q = stats.t.ppf(a, nu)
    scale = np.sqrt((nu - 2) / nu)
    es = stats.t.pdf(q, nu) / (1 - a) * (nu + q**2) / (nu - 1)
    return scale * q, scale * es


def empirical_var_es(losses: np.ndarray, a: float) -> tuple[float, float]:
    var = float(np.quantile(losses, a))
    return var, float(losses[losses >= var].mean())


def nu_from_kurtosis(x: np.ndarray, lo: float = 4.1, hi: float = 30.0) -> float:
    """Method-of-moments Student-t degrees of freedom: excess kurtosis = 6 / (nu - 4)."""
    k = stats.kurtosis(x, fisher=True, bias=False)
    return hi if k <= 6 / (hi - 4) else float(np.clip(4 + 6 / k, lo, hi))


# --- Unconditional models (i.i.d. returns over the window) -------------------------


class HistoricalSimulation(VaRModel):
    name = "Historical"

    def forecast_univariate(self, r, levels):
        return {a: empirical_var_es(-r, a) for a in levels}


class NormalVaR(VaRModel):
    name = "Normal"

    def forecast_univariate(self, r, levels):
        mu, sd = r.mean(), r.std(ddof=1)
        return {a: normal_var_es(mu, sd, a) for a in levels}


class StudentTVaR(VaRModel):
    name = "Student-t"

    def forecast_univariate(self, r, levels):
        mu, sd = r.mean(), r.std(ddof=1)
        nu = nu_from_kurtosis(r)
        out = {}
        for a in levels:
            v, e = std_t_var_es(nu, a)
            out[a] = (-mu + sd * v, -mu + sd * e)
        return out


class CornishFisherVaR(VaRModel):
    """Normal quantile corrected for the skewness and kurtosis of losses."""

    name = "Cornish-Fisher"

    @staticmethod
    def cf_quantile(u, s: float, k: float):
        z = stats.norm.ppf(u)
        return (z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24
                - (2 * z**3 - 5 * z) * s**2 / 36)

    def forecast_univariate(self, r, levels):
        L = -r
        mu, sd = L.mean(), L.std(ddof=1)
        s = stats.skew(L, bias=False)
        k = stats.kurtosis(L, fisher=True, bias=False)
        out = {}
        for a in levels:
            var = mu + sd * self.cf_quantile(a, s, k)
            u = a + (1 - a) * (np.arange(200) + 0.5) / 200  # ES = average tail quantile
            es = mu + sd * float(np.mean(self.cf_quantile(u, s, k)))
            out[a] = (float(var), max(es, float(var)))
        return out


# --- Conditional volatility models -------------------------------------------------


class EWMAVaR(VaRModel):
    """RiskMetrics (1996): exponentially weighted variance, lambda = 0.94, normal shocks."""

    name = "EWMA"

    def __init__(self, lam: float = 0.94):
        self.lam = lam

    def sigma_next(self, r: np.ndarray) -> float:
        T = len(r)
        weights = (1 - self.lam) * self.lam ** np.arange(T)[::-1]
        var = weights @ r**2 + self.lam**T * r.var()
        return float(np.sqrt(var))

    def forecast_univariate(self, r, levels):
        sigma = self.sigma_next(r)
        return {a: normal_var_es(0.0, sigma, a) for a in levels}


def garch_filter(e: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Conditional variances sigma2[0..T]; the last entry is the one-step forecast."""
    s0 = e.var()
    x = omega + alpha * e**2
    tail, _ = lfilter([1.0], [1.0, -beta], x, zi=[beta * s0])
    return np.concatenate([[s0], tail])


def fit_garch_t(r_pct: np.ndarray) -> dict[str, float]:
    """Maximum-likelihood GARCH(1,1) with Student-t errors (returns in percent)."""
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t",
                         rescale=False).fit(disp="off", show_warning=False)
    p = res.params
    return {"mu": p["mu"], "omega": p["omega"], "alpha": p["alpha[1]"],
            "beta": p["beta[1]"], "nu": p["nu"]}


class GARCHtVaR(VaRModel):
    """GARCH(1,1)-t. Parameters are re-estimated every ``refit_every`` days and the
    variance is filtered daily in between, as a production engine would do."""

    name = "GARCH-t"

    def __init__(self, refit_every: int = 20):
        self.refit_every = refit_every
        self._calls = 0
        self.params: dict[str, float] | None = None

    def _filtered(self, r: np.ndarray):
        if self.params is None or self._calls % self.refit_every == 0:
            self.params = fit_garch_t(100 * r)
        self._calls += 1
        p = self.params
        e = 100 * r - p["mu"]
        s2 = garch_filter(e, p["omega"], p["alpha"], p["beta"])
        return e, np.sqrt(s2), p

    def forecast_univariate(self, r, levels):
        _, sigma, p = self._filtered(r)
        out = {}
        for a in levels:
            v, es = std_t_var_es(p["nu"], a)
            out[a] = ((sigma[-1] * v - p["mu"]) / 100, (sigma[-1] * es - p["mu"]) / 100)
        return out


class FilteredHistoricalSimulation(GARCHtVaR):
    """Barone-Adesi et al. (1999): GARCH volatility x empirical standardised residuals."""

    name = "FHS"

    def forecast_univariate(self, r, levels):
        e, sigma, p = self._filtered(r)
        z_loss = -e / sigma[:-1]
        out = {}
        for a in levels:
            v, es = empirical_var_es(z_loss, a)
            out[a] = ((sigma[-1] * v - p["mu"]) / 100, (sigma[-1] * es - p["mu"]) / 100)
        return out


class MonteCarloT(VaRModel):
    """Asset-level Monte Carlo: multivariate Student-t with EWMA covariance.

    Simulating every asset (rather than the portfolio return) is what lets an
    engine add non-linear instruments later by full revaluation.
    """

    name = "Monte Carlo-t"
    multivariate = True

    def __init__(self, n_sims: int = 10_000, lam: float = 0.94, seed: int = 0):
        self.n_sims = n_sims
        self.lam = lam
        self.rng = np.random.default_rng(seed)

    def ewma_cov(self, R: np.ndarray) -> np.ndarray:
        T = len(R)
        wts = (1 - self.lam) * self.lam ** np.arange(T)[::-1]
        Rc = R - R.mean(axis=0)
        return (Rc * wts[:, None]).T @ Rc + self.lam**T * np.cov(Rc, rowvar=False)

    def forecast(self, R, w, levels):
        R, w = np.asarray(R), np.asarray(w)
        cov = self.ewma_cov(R)
        nu = nu_from_kurtosis(R @ w)
        chol = np.linalg.cholesky(cov + 1e-12 * np.eye(len(w)))
        z = self.rng.standard_normal((self.n_sims, len(w))) @ chol.T
        scale = np.sqrt((nu - 2) / self.rng.chisquare(nu, self.n_sims))
        losses = -(z * scale[:, None]) @ w
        return {a: empirical_var_es(losses, a) for a in levels}


def default_models(refit_every: int = 20, mc_sims: int = 10_000) -> list[VaRModel]:
    return [
        HistoricalSimulation(),
        NormalVaR(),
        StudentTVaR(),
        CornishFisherVaR(),
        EWMAVaR(),
        GARCHtVaR(refit_every),
        FilteredHistoricalSimulation(refit_every),
        MonteCarloT(mc_sims),
    ]
