import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

from marketrisk.compare import diebold_mariano, fz0_loss, quantile_loss

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(2)


def test_quantile_loss_is_minimised_at_the_true_quantile():
    y = rng.normal(0, 1, 400_000)
    true_var = -stats.norm.ppf(0.01)
    losses = {k: quantile_loss(y, np.full_like(y, true_var * k), 0.99).mean()
              for k in (0.8, 1.0, 1.2)}
    assert losses[1.0] < losses[0.8] and losses[1.0] < losses[1.2]


def test_fz0_loss_is_minimised_at_true_var_and_es():
    a = 0.975
    y = rng.normal(0, 1, 400_000)
    v = stats.norm.ppf(a)
    e = stats.norm.pdf(v) / (1 - a)
    best = fz0_loss(y, np.full_like(y, v), np.full_like(y, e), a).mean()
    for kv, ke in [(0.8, 0.8), (1.2, 1.2), (1.0, 0.8), (1.0, 1.25)]:
        other = fz0_loss(y, np.full_like(y, v * kv), np.full_like(y, e * ke), a).mean()
        assert best < other


def test_diebold_mariano():
    l1 = rng.normal(1.0, 1.0, 3000)
    assert diebold_mariano(l1, l1)["p_value"] == 1.0
    worse = l1 + 0.2 + rng.normal(0, 0.5, 3000)
    out = diebold_mariano(l1, worse)
    assert out["dm"] < 0 and out["p_value"] < 0.001


def test_full_pipeline_on_synthetic_data(tmp_path):
    spec = importlib.util.spec_from_file_location("run_backtest", ROOT / "scripts/run_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res, summary, rank = mod.run("synthetic", tmp_path, window=250, refit_every=50,
                                 mc_sims=2000, n_synthetic=900, progress=False)
    assert len(summary) == 8 and set(rank["rank"]) == set(range(1, 9))
    assert (tmp_path / "backtest_report.md").exists()
    assert (tmp_path / "figures/traffic_light.png").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["source"] == "synthetic"
