import numpy as np
import pytest
from scipy import integrate, stats

from marketrisk.data import simulate_garch_t
from marketrisk.models import (
    CornishFisherVaR,
    EWMAVaR,
    HistoricalSimulation,
    NormalVaR,
    StudentTVaR,
    default_models,
    fit_garch_t,
    garch_filter,
    std_t_var_es,
)

LEVELS = (0.99, 0.975)
rng = np.random.default_rng(0)


def test_normal_var_es_closed_form():
    r = rng.normal(0.0005, 0.01, 1000)
    mu, sd = r.mean(), r.std(ddof=1)
    var, es = NormalVaR().forecast_univariate(r, [0.99])[0.99]
    assert var == pytest.approx(-mu + sd * stats.norm.ppf(0.99))
    assert es == pytest.approx(-mu + sd * stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01)


def test_historical_simulation_on_known_sample():
    losses = np.arange(1, 1001) / 1000.0  # 0.001 ... 1.000
    var, es = HistoricalSimulation().forecast_univariate(-losses, [0.99])[0.99]
    assert var == pytest.approx(np.quantile(losses, 0.99))
    assert es == pytest.approx(losses[losses >= var].mean())


@pytest.mark.parametrize("nu", [4.5, 6.0, 10.0])
def test_student_t_es_matches_numerical_integration(nu):
    a = 0.975
    scale = np.sqrt((nu - 2) / nu)
    var, es = std_t_var_es(nu, a)
    tail, _ = integrate.quad(lambda u: scale * stats.t.ppf(u, nu), a, 1, limit=200)
    assert var == pytest.approx(scale * stats.t.ppf(a, nu))
    assert es == pytest.approx(tail / (1 - a), rel=1e-4)


def test_cornish_fisher_reduces_to_normal_without_skew_or_kurtosis():
    z = stats.norm.ppf(0.99)
    assert CornishFisherVaR.cf_quantile(0.99, 0.0, 0.0) == pytest.approx(z)


def test_ewma_weights_recent_shocks_more():
    calm = rng.normal(0, 0.01, 500)
    shocked = calm.copy()
    shocked[-5:] = 0.05
    m = EWMAVaR()
    assert m.sigma_next(shocked) > 2 * m.sigma_next(calm)


def test_es_is_at_least_var_for_every_model():
    R = rng.standard_t(4, size=(500, 3)) * 0.01
    w = np.array([0.5, 0.3, 0.2])
    for m in default_models(mc_sims=5000):
        for var, es in m.forecast(R, w, LEVELS).values():
            assert es >= var - 1e-12, m.name
            assert var > 0, m.name
        assert m.forecast(R, w, LEVELS)[0.99][0] > m.forecast(R, w, LEVELS)[0.975][0]


def test_garch_mle_recovers_true_parameters():
    r = simulate_garch_t(6000, omega=0.02, alpha=0.08, beta=0.90, nu=6.0, mu=0.03, seed=3)
    p = fit_garch_t(r)
    assert p["alpha"] == pytest.approx(0.08, abs=0.03)
    assert p["beta"] == pytest.approx(0.90, abs=0.03)
    assert p["alpha"] + p["beta"] == pytest.approx(0.98, abs=0.015)
    assert p["nu"] == pytest.approx(6.0, abs=2.0)


def test_garch_filter_matches_explicit_recursion():
    e = rng.normal(size=50)
    s2 = garch_filter(e, 0.05, 0.1, 0.85)
    ref = [e.var()]
    for x in e:
        ref.append(0.05 + 0.1 * x**2 + 0.85 * ref[-1])
    assert np.allclose(s2, ref)


def test_student_t_needs_fat_tails_to_differ_from_normal():
    r = rng.normal(0, 0.01, 5000)
    v_n = NormalVaR().forecast_univariate(r, [0.99])[0.99][0]
    v_t = StudentTVaR().forecast_univariate(r, [0.99])[0.99][0]
    assert v_t == pytest.approx(v_n, rel=0.05)
