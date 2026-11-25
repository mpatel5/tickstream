from datetime import datetime, timedelta, timezone

import numpy as np

from tickstream.bars import align, regular_hours
from tickstream.events import Bar
from tickstream.research import backtest_oos, rank_pairs
from tickstream.synth import cointegrated_pair


def _bars(n_days=40, seed=0):
    """5-min regular-hours bars: a planted cointegrated pair (A, B), a stock
    correlated with A day to day but not cointegrated (C), and noise (D)."""
    n = n_days * 78
    x, y = cointegrated_pair(n=n, beta=1.0, alpha=0.0, vol=0.002, spread_vol=0.0015,
                             reversion=0.05, seed=seed)
    rng = np.random.default_rng(seed + 1000)  # independent of cointegrated_pair's draws
    c = x + np.cumsum(rng.normal(0, 0.0015, n))   # shares A's moves, drifts away
    d = 4.0 + np.cumsum(rng.normal(0, 0.002, n))
    out, k = [], 0
    for day in range(n_days):
        open_utc = datetime(2026, 9, 1, 13, 30, tzinfo=timezone.utc) + timedelta(days=day)
        for m in range(78):
            ts = open_utc + timedelta(minutes=5 * m)
            for sym, s in (("A", x), ("B", y), ("C", c), ("D", d)):
                px = float(np.exp(s[k]))
                out.append(Bar(sym, ts, px, px, px, px, 100))
            k += 1
        # an extended-hours bar that regular_hours must drop
        out.append(Bar("A", open_utc - timedelta(hours=1), 1.0, 1.0, 1.0, 1.0, 1))
    return out


def test_align_and_regular_hours():
    panel = align(regular_hours(_bars(n_days=3)))
    assert panel.symbols == ["A", "B", "C", "D"]
    assert panel.close.shape == (4, 3 * 78)
    assert panel.returns().shape == (4, 3 * 77)   # overnight returns dropped


def test_align_forward_fills_gaps():
    t0 = datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc)
    bars = [Bar("A", t0 + timedelta(minutes=5 * i), 10 + i, 0, 0, 10 + i, 1) for i in range(4)]
    bars += [Bar("B", t0 + timedelta(minutes=5 * i), 20, 0, 0, 20 + i, 1) for i in (0, 2, 3)]
    p = align(bars)
    assert p.close[1].tolist() == [20, 20, 22, 23]


def test_pipeline_finds_planted_pair_and_trades_it_out_of_sample():
    panel = align(regular_hours(_bars()))
    split = int(len(panel.times) * 2 / 3)
    pairs, n_tested = rank_pairs(panel.slice(0, split), corr_threshold=0.3)
    names = [{p.a, p.b} for p in pairs]
    assert n_tested >= 1
    assert {"A", "B"} in names
    assert all("C" not in n for n in names)       # correlated, not cointegrated
    assert all("D" not in n for n in names)
    assert pairs[0].coint.significance == 0.01
    best = next(p for p in pairs if {p.a, p.b} == {"A", "B"})
    r = backtest_oos(panel, split, best, halflife=390, periods_per_year=252 * 78)
    assert r.trades > 0
    assert r.total_return > 0
