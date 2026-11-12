"""Market data: Yahoo Finance prices (cached) and a simulated portfolio for offline use.

Default universe: 10 liquid US-listed ETFs across asset classes, all trading
since 2004, so a 500-day estimation window still leaves 2008 in the back-test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKERS = {
    "SPY": "US large cap equity",
    "QQQ": "US tech equity",
    "IWM": "US small cap equity",
    "EFA": "Developed ex-US equity",
    "EEM": "Emerging markets equity",
    "VNQ": "US real estate (REITs)",
    "TLT": "US Treasuries 20y+",
    "IEF": "US Treasuries 7-10y",
    "LQD": "US investment-grade credit",
    "GLD": "Gold",
}
DEFAULT_START = "2005-01-01"
DEFAULT_END = "2025-12-31"


def load_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: str | Path | None = "data/raw",
) -> pd.DataFrame:
    """Adjusted close prices (dividends reinvested), cached as Parquet."""
    tickers = list(tickers or DEFAULT_TICKERS)
    cache = None
    if cache_dir is not None:
        key = f"prices_{'-'.join(sorted(tickers))}_{start}_{end}.parquet"
        cache = Path(cache_dir) / key
        if cache.exists():
            return pd.read_parquet(cache)[tickers]

    import yfinance as yf  # optional dependency: pip install -e ".[data]"

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"][tickers].dropna(how="any")
    if prices.empty:
        raise RuntimeError("No data returned by Yahoo Finance (network or ticker issue).")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache)
    return prices


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().dropna(how="any")


def portfolio_returns(asset_returns: pd.DataFrame, weights: np.ndarray | None = None) -> pd.Series:
    """Daily-rebalanced portfolio return w'r (equal weight by default)."""
    w = equal_weights(asset_returns.shape[1]) if weights is None else np.asarray(weights, float)
    return pd.Series(asset_returns.to_numpy() @ w, index=asset_returns.index, name="portfolio")


def equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def simulate_garch_t(
    n: int,
    omega: float = 0.02,
    alpha: float = 0.08,
    beta: float = 0.90,
    nu: float = 6.0,
    mu: float = 0.03,
    seed: int | None = 0,
) -> np.ndarray:
    """Simulate a GARCH(1,1) path with standardised Student-t shocks (returns in %)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    var = np.empty(n)
    r = np.empty(n)
    var[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t > 0:
            var[t] = omega + alpha * (r[t - 1] - mu) ** 2 + beta * var[t - 1]
        r[t] = mu + np.sqrt(var[t]) * z[t]
    return r


def make_synthetic(n_days: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Simulated daily returns for 10 assets with volatility clustering and fat tails.

    One common GARCH-t market factor drives the equities; bonds load negatively
    on it (flight to quality); gold is nearly independent. Asset names mirror the
    default ETF universe so the rest of the pipeline is unchanged.
    """
    rng = np.random.default_rng(seed)
    market = simulate_garch_t(n_days, seed=seed) / 100
    rates = simulate_garch_t(n_days, omega=0.01, alpha=0.05, beta=0.93, nu=8, mu=0.0,
                             seed=seed + 1) / 100 * 0.6
    betas = {"SPY": 1.0, "QQQ": 1.2, "IWM": 1.15, "EFA": 0.9, "EEM": 1.1, "VNQ": 1.1,
             "TLT": -0.25, "IEF": -0.1, "LQD": 0.1, "GLD": 0.05}
    rate_load = {"TLT": 2.2, "IEF": 1.0, "LQD": 0.8}
    idio_vol = {"SPY": 0.002, "QQQ": 0.006, "IWM": 0.006, "EFA": 0.005, "EEM": 0.008,
                "VNQ": 0.008, "TLT": 0.003, "IEF": 0.001, "LQD": 0.002, "GLD": 0.010}
    out = {}
    for k, b in betas.items():
        eps = rng.standard_t(5, n_days) * np.sqrt(3 / 5) * idio_vol[k]
        out[k] = b * market + rate_load.get(k, 0.0) * rates + eps
    idx = pd.bdate_range("2010-01-04", periods=n_days)
    return pd.DataFrame(out, index=idx)
