import numpy as np
import pytest

from tickstream.coint import adf_tstat, critical_values, engle_granger, half_life
from tickstream.synth import cointegrated_pair, independent_pair

sm = pytest.importorskip("statsmodels.tsa.stattools")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_statsmodels_coint(seed):
    x, y = cointegrated_pair(n=3_000, seed=seed, reversion=0.01)
    ours = engle_granger(y, x)
    t, _, crit = sm.coint(y, x, trend="c", autolag="aic")
    assert ours.t_stat == pytest.approx(t, rel=1e-6)
    assert [ours.crit[l] for l in (0.01, 0.05, 0.10)] == pytest.approx(list(crit), rel=1e-4)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_adf_fixed_lag_matches_statsmodels():
    rng = np.random.default_rng(3)
    e = np.zeros(2_000)
    for t in range(1, len(e)):
        e[t] = 0.97 * e[t - 1] + rng.normal()
    ours, p = adf_tstat(e, maxlag=4, autolag=False)
    theirs = sm.adfuller(e, maxlag=4, autolag=None, regression="n")[0]
    assert p == 4
    assert ours == pytest.approx(theirs, rel=1e-6)


def test_separates_cointegrated_from_independent():
    x, y = cointegrated_pair(n=5_000, beta=1.3, seed=4)
    good = engle_granger(y, x)
    assert good.significance == 0.01
    assert good.beta == pytest.approx(1.3, abs=0.05)
    x2, y2 = independent_pair(n=5_000, seed=5)
    assert engle_granger(y2, x2).significance is None


def test_half_life_of_ar1():
    rng = np.random.default_rng(6)
    phi = 0.95  # true half-life = ln(0.5)/ln(0.95) ~ 13.5 bars
    s = np.zeros(50_000)
    for t in range(1, len(s)):
        s[t] = phi * s[t - 1] + rng.normal()
    assert half_life(s) == pytest.approx(np.log(0.5) / np.log(phi), rel=0.1)


def test_critical_values_shrink_toward_asymptote():
    small, large = critical_values(100), critical_values(100_000)
    assert small[0.05] < large[0.05] < -3.33
