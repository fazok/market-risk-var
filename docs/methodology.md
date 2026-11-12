# Methodology

## 1. Data and portfolio

Ten US-listed ETFs, all trading since 2004: SPY, QQQ, IWM (US equity), EFA, EEM
(international equity), VNQ (REITs), TLT, IEF (Treasuries), LQD (IG credit),
GLD (gold). Daily adjusted closes from Yahoo Finance, 2005 to 2025, with
dividends reinvested. The portfolio is equal-weight and rebalanced daily, so its
return is `r_p = w'r`. The loss is `L = -r_p`, and VaR and ES are positive
fractions of portfolio value.

HYG (high yield) was considered but only starts in 2007, which would have
removed 2008 from the out-of-sample period.

## 2. Models

Every model forecasts day *t* using only the 500 trading days before *t*.

| Model | VaR / ES | Key assumption |
|---|---|---|
| Historical simulation | Empirical quantile / tail mean of the 500 losses | i.i.d. returns, history repeats |
| Normal | `-mu + sigma z_a`, `-mu + sigma phi(z_a) / (1 - a)` | i.i.d. Gaussian |
| Student-t | Unit-variance t quantile and closed-form ES | i.i.d. fat tails; nu by method of moments, `nu = 4 + 6 / excess kurtosis` |
| Cornish-Fisher | Normal quantile corrected for skewness and kurtosis; ES as the average of tail quantiles | Moment expansion |
| EWMA | `sigma^2_{t+1} = lambda sigma^2_t + (1 - lambda) r_t^2`, lambda = 0.94, normal | RiskMetrics (1996) |
| GARCH(1,1)-t | MLE of `(mu, omega, alpha, beta, nu)` every 20 days, daily filtering in between | Volatility clustering, fat-tailed shocks |
| FHS | GARCH volatility forecast x empirical quantile of standardised residuals | Barone-Adesi et al. (1999): GARCH dynamics, non-parametric tails |
| Monte Carlo-t | 10,000 draws of a multivariate t with EWMA covariance, at asset level | Joint fat tails; extensible to options by full revaluation |

## 3. Back-tests

* **Kupiec (1995) POF.** `LR = -2 ln[(1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x)]`, chi2(1).
* **Christoffersen (1998).** A first-order Markov test that exceptions are
  independent (chi2(1)). Conditional coverage = POF + independence (chi2(2)).
* **Basel traffic light.** Rolling 250-day exception count at 99%: green 0–4,
  yellow 5–9 (plus-factor 0.40–0.85), red 10+.
* **Acerbi & Székely (2014) Z2.**
  `Z2 = sum(L_t 1{L_t > VaR_t} / ES_t) / (T (1 - a)) - 1`. It is zero in
  expectation for a correct ES. The fixed thresholds 0.70 / 1.80 follow the
  authors' finding that the critical values barely depend on the distribution.
* **Square-root-of-time.** The 1-day VaR x sqrt(10), issued on day *t*, is
  compared with the compounded loss over days *t* to *t*+9, on non-overlapping
  blocks.

## 4. Model comparison

A back-test can reject a model, but it cannot say which of two models that both
pass is better. That needs a **consistent scoring function**: one whose expected
value is minimised by the true risk measure.

* VaR: the quantile ("tick") loss.
* (VaR, ES) jointly: ES alone is not elicitable (Gneiting 2011), but the pair is
  (Fissler & Ziegel 2016). The **FZ0** loss of Patton, Ziegel & Chen (2019) is
  used, with `v = -VaR`, `e = -ES` and tau = 2.5%:
  `FZ0 = -1{y <= v}(v - y) / (tau e) + v / e + ln(-e) - 1`.
* **Diebold-Mariano (1995)** on loss differentials, with a Newey-West long-run
  variance (Bartlett kernel, lag `floor(4 (T/100)^(2/9))`).

Both losses are unit-tested to be minimised at the true VaR/ES on simulated
data.

## 5. Discussion

* The 500-day unconditional models are the slowest to react. That is exactly
  the pro-cyclicality problem behind Basel 2.5's stressed VaR.
* Even the best conditional models fail conditional coverage over 19 years. A
  single GARCH(1,1) cannot capture regime shifts like the 2022 joint sell-off of
  bonds and equities.
* Cornish-Fisher shows why unconditional coverage alone is a weak test: a model
  can be right on average and wrong exactly when it matters.

## 6. Extensions (research directions)

1. **Model Confidence Set** (Hansen, Lunde & Nason 2011) instead of pairwise DM tests.
2. **Asymmetric and regime models:** GJR-GARCH, EGARCH, Markov-switching
   volatility, and realised-volatility models (HAR) with intraday data.
3. **Copulas** for the multivariate Monte Carlo (t vs Clayton tail dependence).
4. **Non-linear portfolios:** add an options sleeve and compare full revaluation
   with delta-gamma approximations.
5. **FRTB:** liquidity-horizon-scaled ES and the stressed-period calibration.

## References

* Acerbi, C., Székely, B. (2014). Back-testing expected shortfall. *Risk*, December.
* Barone-Adesi, G., Giannopoulos, K., Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5).
* Basel Committee on Banking Supervision (1996). *Supervisory framework for the use of backtesting in conjunction with the internal models approach.*
* Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk* (FRTB).
* Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4).
* Diebold, F., Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3).
* Fissler, T., Ziegel, J. (2016). Higher order elicitability and Osband's principle. *Annals of Statistics*, 44(4).
* Gneiting, T. (2011). Making and evaluating point forecasts. *JASA*, 106(494).
* J.P. Morgan / Reuters (1996). *RiskMetrics Technical Document*, 4th ed.
* Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2).
* Patton, A., Ziegel, J., Chen, R. (2019). Dynamic semiparametric models for expected shortfall (and Value-at-Risk). *Journal of Econometrics*, 211(2).
