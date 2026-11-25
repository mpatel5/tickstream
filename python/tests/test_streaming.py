import numpy as np

from tickstream import KLL, MisraGries
from tickstream.screen import exact_pairs, screen
from tickstream.synth import factor_returns, zipf_trades


def test_kll_rank_error_small():
    rng = np.random.default_rng(1)
    data = rng.standard_t(3, size=100_000)
    s = KLL(k=200, seed=1)
    s.extend(data)
    srt = np.sort(data)
    for q in (0.01, 0.05, 0.5, 0.95, 0.99):
        true_rank = np.searchsorted(srt, s.quantile(q), side="right") / len(data)
        assert abs(true_rank - q) < 0.02
    assert s.stored_items < 1000  # vs 100k raw


def test_kll_merge():
    rng = np.random.default_rng(2)
    a, b = KLL(k=200, seed=1), KLL(k=200, seed=2)
    xa, xb = rng.normal(size=50_000), rng.normal(2, 1, size=50_000)
    a.extend(xa)
    b.extend(xb)
    a.merge(b)
    allx = np.sort(np.concatenate([xa, xb]))
    assert abs(np.searchsorted(allx, a.quantile(0.5)) / len(allx) - 0.5) < 0.02
    assert a.n == 100_000


def test_misra_gries_finds_heavy_hitters():
    trades = zipf_trades(50_000, seed=4)
    mg = MisraGries(k=50)
    for t in trades:
        mg.update(int(t))
    true = np.bincount(trades)
    heavy = set(np.where(true >= 0.02 * len(trades))[0].tolist())
    assert heavy <= {k for k, _ in mg.heavy_hitters(0.02)}
    for k in heavy:
        assert true[k] - mg.estimate(k) <= mg.error_bound


def test_screen_no_false_positives_high_recall():
    X = factor_returns(400, 390, seed=7, n_twins=20)
    truth = exact_pairs(X, 0.8)
    found = {(i, j) for i, j, _ in screen(X, 0.8, k=64, slack=0.15).pairs}
    assert truth and found <= truth
    assert len(found) / len(truth) >= 0.95
