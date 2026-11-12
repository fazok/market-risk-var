"""Figures for the README and the back-testing report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .backtest import BacktestResult, rolling_exceptions  # noqa: E402

NAVY, TEAL, AMBER, RED, GREY = "#1f3a5f", "#2a9d8f", "#e9a23b", "#c0392b", "#8a8f98"
PALETTE = [NAVY, TEAL, AMBER, RED, "#6c5b7b", "#3d8bd9", "#7f8c3a", GREY]
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 9,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def var_bands(res: BacktestResult, models: list[str], path: Path, level: float = 0.99,
              start: str | None = None, end: str | None = None, title: str = "") -> Path:
    loss = res.losses.loc[start:end]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(loss.index, 100 * loss, width=1.0, color=GREY, alpha=0.6, label="Daily loss")
    for m, c in zip(models, [NAVY, TEAL, AMBER, RED]):
        v = res.var[level][m].loc[start:end]
        ax.plot(v.index, 100 * v, color=c, lw=1.2, label=f"{m} VaR {level:.0%}")
        hit = loss > v
        ax.scatter(loss.index[hit], 100 * loss[hit], color=c, s=10, zorder=3)
    ax.set(ylabel="% of portfolio", title=title or f"1-day VaR {level:.0%} and exceptions")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=len(models) + 1, fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y" if start is None else "%b %Y"))
    return _save(fig, path)


def traffic_light_map(res: BacktestResult, path: Path, level: float = 0.99) -> Path:
    counts = rolling_exceptions(res.exceptions(level))
    zones = np.select([counts <= 4, counts <= 9], [0, 1], default=2).T
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(counts.columns) + 1.0))
    ax.imshow(zones, aspect="auto", interpolation="nearest",
              cmap=ListedColormap([TEAL, AMBER, RED]), vmin=0, vmax=2,
              extent=[mdates.date2num(counts.index[0]), mdates.date2num(counts.index[-1]),
                      len(counts.columns) - 0.5, -0.5])
    ax.set_yticks(range(len(counts.columns)))
    ax.set_yticklabels(counts.columns)
    ax.xaxis_date()
    ax.grid(False)
    ax.set_title("Basel traffic light over time (exceptions in last 250 days: "
                 "green 0-4, amber 5-9, red 10+)")
    return _save(fig, path)


def ranking_plot(rank: pd.DataFrame, path: Path) -> Path:
    r = rank.sort_values("fz0_loss", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sig = r["dm_p_value"] < 0.05
    colors = [TEAL if not s else GREY for s in sig]
    colors[-1] = NAVY
    gap = r["fz0_loss"] - r["fz0_loss"].min()
    ax.barh(r.index, gap + 1e-4, color=colors)
    ax.text(gap.max() * 0.01, len(r) - 1, "  best model", va="center", color=NAVY,
            fontweight="bold")
    ax.set(xlabel="Average FZ0 loss minus best (lower is better)",
           title="Joint VaR-ES ranking (grey = significantly worse than best, DM 5%)")
    return _save(fig, path)


def cumulative_returns(asset_returns: pd.DataFrame, port: pd.Series, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for c, color in zip(asset_returns.columns, PALETTE * 2):
        ax.plot((1 + asset_returns[c]).cumprod(), lw=0.7, alpha=0.5, color=color, label=c)
    ax.plot((1 + port).cumprod(), lw=2, color="black", label="Equal-weight portfolio")
    ax.set_yscale("log")
    ax.set(title="Growth of 1 (log scale)", ylabel="")
    ax.legend(ncol=6, fontsize=7)
    return _save(fig, path)
