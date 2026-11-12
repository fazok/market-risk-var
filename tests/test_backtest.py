import numpy as np
import pandas as pd
import pytest

from marketrisk.backtest import (
    acerbi_szekely_z2,
    christoffersen,
    kupiec_pof,
    plus_factor,
    run_rolling,
    traffic_light_zone,
)
from marketrisk.data import simulate_garch_t
from marketrisk.models import GARCHtVaR, NormalVaR

rng = np.random.default_rng(1)


def test_kupiec_reference_value():
    # 10 exceptions in 250 days at 99%: LR = 12.96 (Jorion, Value at Risk).
    k = kupiec_pof(250, 10, 0.01)
    assert k["lr"] == pytest.approx(12.9555, abs=1e-3)
    assert k["p_value"] < 0.001
    assert kupiec_pof(250, 2, 0.01)["p_value"] > 0.5


def test_kupiec_handles_zero_exceptions():
    k = kupiec_pof(250, 0, 0.01)
    assert np.isfinite(k["lr"]) and k["lr"] == pytest.approx(-2 * 250 * np.log(0.99))


@pytest.mark.parametrize("x,zone,factor", [(0, "green", 0.0), (4, "green", 0.0),
                                           (5, "yellow", 0.40), (9, "yellow", 0.85),
                                           (10, "red", 1.0), (25, "red", 1.0)])
def test_traffic_light_boundaries(x, zone, factor):
    assert traffic_light_zone(x) == zone
    assert plus_factor(x) == factor


def test_christoffersen_detects_clustering():
    iid = (rng.random(2500) < 0.01).astype(int)
    clustered = np.zeros(2500, dtype=int)
    for start in rng.choice(np.arange(0, 2490, 50), 5, replace=False):
        clustered[start:start + 5] = 1  # 25 exceptions in 5 bursts
    assert christoffersen(iid, 0.01)["p_ind"] > 0.05
    assert christoffersen(clustered, 0.01)["p_ind"] < 0.001


def test_z2_is_near_zero_for_correct_es_and_positive_when_underestimated():
    from scipy import stats

    a = 0.975
    L = rng.normal(0, 1, 200_000)
    var = np.full_like(L, stats.norm.ppf(a))
    es = np.full_like(L, stats.norm.pdf(stats.norm.ppf(a)) / (1 - a))
    assert abs(acerbi_szekely_z2(L, var, es, a)) < 0.03
    assert acerbi_szekely_z2(L, var * 0.7, es * 0.7, a) > 0.70


def _garch_returns(n=2600, seed=5):
    r = simulate_garch_t(n, omega=0.02, alpha=0.08, beta=0.90, nu=5.0, mu=0.0, seed=seed) / 100
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"x": r}, index=idx)


def test_correct_model_passes_and_normal_iid_fails_on_garch_data():
    R = _garch_returns()
    res = run_rolling(R, np.array([1.0]), [GARCHtVaR(refit_every=50), NormalVaR()],
                      window=500, levels=(0.99,))
    hits = res.exceptions(0.99)
    n = len(hits)
    garch = kupiec_pof(n, int(hits["GARCH-t"].sum()), 0.01)
    normal = kupiec_pof(n, int(hits["Normal"].sum()), 0.01)
    assert garch["p_value"] > 0.01
    assert normal["exceptions"] > garch["exceptions"]
    assert christoffersen(hits["Normal"].to_numpy(), 0.01)["p_cc"] < 0.05
