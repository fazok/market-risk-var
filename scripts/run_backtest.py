"""Rolling VaR / ES back-test of all models, with report, figures and metrics.

Usage:
    python scripts/run_backtest.py --source yahoo       # 10 ETFs, 2005-2025
    python scripts/run_backtest.py --source synthetic   # simulated, no network
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from marketrisk import data as D
from marketrisk import plots
from marketrisk.backtest import (
    STRESS_PERIODS,
    period_exceptions,
    run_rolling,
    summary_table,
    ten_day_scaling,
)
from marketrisk.compare import dm_matrix, loss_table, ranking
from marketrisk.models import default_models
from marketrisk.report import write_report

ROOT = Path(__file__).resolve().parents[1]


def run(source: str, out_dir: Path, window: int = 500, refit_every: int = 20,
        mc_sims: int = 10_000, n_synthetic: int = 3000, weights=None, progress: bool = True):
    if source == "yahoo":
        asset_r = D.to_returns(D.load_prices(cache_dir=ROOT / "data/raw"))
        assets = f"{asset_r.shape[1]} ETFs ({', '.join(asset_r.columns)}), equal weight"
    else:
        asset_r = D.make_synthetic(n_synthetic)
        assets = f"{asset_r.shape[1]} simulated assets, equal weight"
    w = D.equal_weights(asset_r.shape[1]) if weights is None else np.asarray(weights)
    print(f"[data] {source}: {len(asset_r):,} days x {asset_r.shape[1]} assets")

    t0 = time.time()
    models = default_models(refit_every=refit_every, mc_sims=mc_sims)
    res = run_rolling(asset_r, w, models, window=window, progress=progress)
    print(f"[backtest] {len(res.returns):,} forecasts x {len(models)} models "
          f"in {time.time() - t0:.0f}s")

    summary = summary_table(res)
    periods = period_exceptions(res, STRESS_PERIODS)
    losses = loss_table(res)
    rank = ranking(losses)
    ten = ten_day_scaling(res)

    fig = out_dir / "figures"
    plots.cumulative_returns(asset_r, D.portfolio_returns(asset_r, w), fig / "portfolio.png")
    plots.traffic_light_map(res, fig / "traffic_light.png")
    plots.ranking_plot(rank, fig / "ranking.png")
    focus = ["Historical", "Normal", "FHS"]
    plots.var_bands(res, focus, fig / "var_full.png", title="1-day VaR 99%, full period")
    first = str(res.returns.index[0].date())
    gfc = ("2008-06-01", "2009-06-30") if first < "2008-06-01" else (None, None)
    covid = ("2019-12-01", "2020-08-31") if first < "2019-12-01" else (None, None)
    if source == "synthetic":  # no real crises: zoom on the most volatile year instead
        vol = res.returns.rolling(250).std()
        end = vol.idxmax()
        gfc = covid = (str((end - np.timedelta64(365, "D")).date()), str(end.date()))
    plots.var_bands(res, focus, fig / "var_gfc.png", start=gfc[0], end=gfc[1],
                    title="Global financial crisis")
    plots.var_bands(res, focus, fig / "var_covid.png", start=covid[0], end=covid[1],
                    title="COVID-19 crash")

    ctx = {
        "source": source, "assets": assets, "window": window, "refit_every": refit_every,
        "start": str(res.returns.index[0].date()), "end": str(res.returns.index[-1].date()),
        "n_days": len(res.returns), "summary": summary, "periods": periods,
        "ranking": rank, "dm_matrix": dm_matrix(losses), "ten_day": ten,
    }
    write_report(out_dir / "backtest_report.md", ctx)
    metrics = {
        "source": source, "period": [ctx["start"], ctx["end"]],
        "summary": summary.reset_index().to_dict(orient="records"),
        "ranking": rank.reset_index().to_dict(orient="records"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(summary[["exceptions", "expected", "kupiec_p", "christ_cc_p", "es_zone"]].round(3))
    print(rank[["rank", "fz0_loss", "dm_p_value"]].round(4))
    print(f"[report] {out_dir / 'backtest_report.md'}")
    return res, summary, rank


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["yahoo", "synthetic"], default="yahoo")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--refit-every", type=int, default=20)
    ap.add_argument("--mc-sims", type=int, default=10_000)
    args = ap.parse_args()
    out = args.out or ROOT / ("reports" if args.source == "yahoo" else "reports/synthetic")
    run(args.source, out, window=args.window, refit_every=args.refit_every,
        mc_sims=args.mc_sims)


if __name__ == "__main__":
    main()
