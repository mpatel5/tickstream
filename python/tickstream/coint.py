"""Engle-Granger cointegration test, implemented with NumPy only.

Two log-price series are cointegrated if some combination y - beta*x - alpha
is stationary: it keeps getting pulled back to its mean instead of wandering
off. Correlation of returns doesn't guarantee that; this test checks it.

Step 1: regress y on x (OLS) to get alpha, beta and the residual spread.
Step 2: run an augmented Dickey-Fuller (ADF) test on the spread. A strongly
negative t-statistic means the spread mean-reverts.

The ADF statistic doesn't follow a t-distribution, so it's compared with
MacKinnon (2010) critical values for two variables with a constant. Results
match statsmodels.tsa.stattools.coint (see python/tests/test_coint.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# MacKinnon (2010), Engle-Granger with constant, N = 2 variables:
# crit(T) = b0 + b1/T + b2/T^2 for the 1%, 5% and 10% levels.
_MACKINNON_EG2 = {
    0.01: (-3.89644, -10.9519, -22.527),
    0.05: (-3.33613, -6.1101, -6.823),
    0.10: (-3.04445, -4.2412, -2.720),
}


def critical_values(nobs: int) -> dict[float, float]:
    return {lvl: b0 + b1 / nobs + b2 / nobs**2 for lvl, (b0, b1, b2) in _MACKINNON_EG2.items()}


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, resid, float(resid @ resid)


def adf_tstat(e: np.ndarray, maxlag: int | None = None, autolag: bool = True) -> tuple[float, int]:
    """ADF t-statistic with no constant (for regression residuals).

    Regresses de[t] on e[t-1] and p lagged differences; returns
    (t-stat on e[t-1], p). With autolag, p minimises AIC over 0..maxlag on a
    common sample, then the chosen model is refit on all usable rows.
    """
    e = np.asarray(e, dtype=float)
    d = np.diff(e)
    n = len(e)
    if maxlag is None:
        maxlag = int(math.ceil(12.0 * (n / 100.0) ** 0.25))
    maxlag = max(0, min(maxlag, n // 2 - 2))

    def design(p: int, start: int) -> tuple[np.ndarray, np.ndarray]:
        rows = np.arange(start, len(d))
        cols = [e[rows]] + [d[rows - i] for i in range(1, p + 1)]
        return d[rows], np.column_stack(cols)

    if autolag:
        y_full, X_full = design(maxlag, maxlag)
        m = len(y_full)
        best_p, best_aic = 0, np.inf
        for p in range(maxlag + 1):
            _, _, ssr = _ols(y_full, X_full[:, : p + 1])
            aic = m * math.log(ssr / m) + 2 * (p + 1)
            if aic < best_aic:
                best_p, best_aic = p, aic
        p = best_p
    else:
        p = maxlag

    y, X = design(p, p)
    coef, _, ssr = _ols(y, X)
    dof = len(y) - X.shape[1]
    cov = (ssr / dof) * np.linalg.inv(X.T @ X)
    return float(coef[0] / math.sqrt(cov[0, 0])), p


def half_life(spread: np.ndarray) -> float:
    """Bars for a deviation of the spread to decay by half (from an AR(1) fit)."""
    s = np.asarray(spread, dtype=float)
    X = np.column_stack([np.ones(len(s) - 1), s[:-1]])
    coef, _, _ = _ols(np.diff(s), X)
    b = coef[1]
    return float(-math.log(2) / math.log1p(b)) if -1 < b < 0 else math.inf


@dataclass
class CointResult:
    t_stat: float            # ADF statistic on the spread; more negative = stronger
    crit: dict[float, float]  # MacKinnon critical values {0.01, 0.05, 0.10}
    alpha: float
    beta: float              # y ~ alpha + beta * x
    half_life: float         # bars
    lags: int
    swapped: bool = False    # True if the better direction regressed x on y

    @property
    def significance(self) -> float | None:
        """Smallest level (0.01 / 0.05 / 0.10) at which the test rejects, else None."""
        for lvl in (0.01, 0.05, 0.10):
            if self.t_stat < self.crit[lvl]:
                return lvl
        return None


def engle_granger(y: np.ndarray, x: np.ndarray, maxlag: int | None = None,
                  autolag: bool = True) -> CointResult:
    """Engle-Granger test of y on x (one direction, like statsmodels' coint)."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    (alpha, beta), resid, _ = _ols(y, X)
    t, lags = adf_tstat(resid, maxlag=maxlag, autolag=autolag)
    return CointResult(t, critical_values(len(y)), float(alpha), float(beta),
                       half_life(resid), lags)


def engle_granger_both(a: np.ndarray, b: np.ndarray, **kw) -> CointResult:
    """Run the test in both directions and keep the stronger one.

    Engle-Granger isn't symmetric. Taking the better of two directions makes the
    test slightly too willing to reject, which is one more reason to treat a
    passing pair as a candidate to verify out of sample, not a finding.
    """
    r1 = engle_granger(a, b, **kw)
    r2 = engle_granger(b, a, **kw)
    if r2.t_stat < r1.t_stat:
        r2.swapped = True
        return r2
    return r1
