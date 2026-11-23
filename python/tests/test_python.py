import numpy as np
import pytest

import tickstream as ts
from tickstream.pairs import backtest
from tickstream.synth import cointegrated_pair, independent_pair


def test_count_min_string_and_int_keys():
    cms = ts.CountMinSketch.from_error(0.001, 0.01)
    for sym, vol in [("AAPL", 300), ("MSFT", 200), ("AAPL", 100), (42, 7)]:
        cms.add(sym, vol)
    assert cms["AAPL"] >= 400
    assert cms["MSFT"] >= 200
    assert cms.estimate(42) >= 7
    assert cms.total == 607


def test_count_min_add_many_matches_loop():
    keys = np.random.default_rng(0).integers(0, 1000, 50_000, dtype=np.uint64)
    a, b = ts.CountMinSketch(2048, 4), ts.CountMinSketch(2048, 4)
    a.add_many(keys)
    for k in keys:
        b.add(int(k))
    assert all(a.estimate(int(k)) == b.estimate(int(k)) for k in range(1000))


def test_hll_distinct_count():
    h = ts.HyperLogLog(14)
    h.add_many(np.arange(200_000, dtype=np.uint64))
    h.add_many(np.arange(200_000, dtype=np.uint64))
    assert abs(h.estimate() - 200_000) / 200_000 < 4 * h.standard_error


def test_reservoir_holds_python_objects():
    r = ts.Reservoir(5, seed=1)
    r.add_many({"px": i} for i in range(1000))
    assert len(r.sample) == 5 and r.seen == 1000
    assert all(isinstance(s, dict) for s in r.sample)


def test_running_stats_matches_numpy():
    xs = np.random.default_rng(1).normal(50, 3, 10_000)
    rs = ts.RunningStats()
    rs.add_many(xs)
    assert rs.mean == pytest.approx(xs.mean())
    assert rs.variance == pytest.approx(xs.var(ddof=1))


def test_pair_stats_recovers_hedge_ratio():
    x, y = cointegrated_pair(n=30_000, beta=1.3, seed=2)
    ps = ts.PairStats(halflife=2_000)
    beta, spread, z = ps.update_many(x, y)
    assert ps.beta == pytest.approx(1.3, abs=0.1)
    assert len(beta) == len(spread) == len(z) == 30_000


def test_update_many_equals_update_loop():
    x, y = cointegrated_pair(n=2_000, seed=3)
    a, b = ts.PairStats(100), ts.PairStats(100)
    _, _, z = a.update_many(x, y)
    z_loop = np.array([b.update(xi, yi)[2] for xi, yi in zip(x, y)])
    np.testing.assert_allclose(z, z_loop)


def test_backtest_trades_and_profits_on_cointegrated_pair():
    x, y = cointegrated_pair(n=40_000, seed=4)
    res = backtest(x, y, halflife=1_000, warmup=2_000)
    assert res.trades > 10
    assert res.total_return > 0
    assert res.equity.shape == (40_000,)


def test_backtest_does_not_find_edge_in_unrelated_walks():
    x, y = independent_pair(n=40_000, seed=5)
    res = backtest(x, y, halflife=1_000, warmup=2_000)
    good = backtest(*cointegrated_pair(n=40_000, seed=5), halflife=1_000, warmup=2_000)
    assert res.sharpe < good.sharpe


def test_backtest_validates_input():
    with pytest.raises(ValueError):
        backtest(np.zeros(10), np.zeros(11))
