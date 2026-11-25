"""Synthetic price data for testing without a market-data feed."""

from __future__ import annotations

import numpy as np


def cointegrated_pair(
    n: int = 20_000,
    beta: float = 1.3,
    alpha: float = 0.2,
    vol: float = 0.01,
    spread_vol: float = 0.004,
    reversion: float = 0.02,
    start: float = 4.0,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Log prices (x, y) where y = alpha + beta * x + s and s is mean-reverting.

    x is a Gaussian random walk; s follows a discrete Ornstein-Uhlenbeck process
    s[t] = (1 - reversion) * s[t-1] + spread_vol * eps[t].
    """
    rng = np.random.default_rng(seed)
    x = start + np.cumsum(rng.normal(0.0, vol, n))
    s = np.empty(n)
    s[0] = 0.0
    shocks = rng.normal(0.0, spread_vol, n)
    for t in range(1, n):
        s[t] = (1.0 - reversion) * s[t - 1] + shocks[t]
    y = alpha + beta * x + s
    return x, y


def independent_pair(n: int = 20_000, vol: float = 0.01, seed: int | None = 1) -> tuple[np.ndarray, np.ndarray]:
    """Two unrelated random walks: a control case with no real relationship."""
    rng = np.random.default_rng(seed)
    return 4.0 + np.cumsum(rng.normal(0, vol, n)), 4.0 + np.cumsum(rng.normal(0, vol, n))


def factor_returns(n_series: int, window: int, n_sectors: int = 20,
                   loading_range=(0.2, 0.9), df: float = 4.0, seed: int = 0,
                   n_twins: int = 0) -> np.ndarray:
    """Cross-section of returns from a sector factor model, shape (n_series, window).

    Market + sector factors with Student-t noise (fat tails). The last
    `n_twins` series are near-copies of others (corr ~0.95), like share
    classes or overlapping ETFs, so the screener has known pairs to find.
    """
    rng = np.random.default_rng(seed)
    market = rng.standard_normal(window)
    sectors = rng.standard_normal((n_sectors, window))
    sector_of = rng.integers(0, n_sectors, size=n_series)
    beta_m = rng.uniform(0.2, 0.6, size=(n_series, 1))
    beta_s = rng.uniform(*loading_range, size=(n_series, 1))
    noise = rng.standard_t(df, size=(n_series, window)) / np.sqrt(df / (df - 2))
    X = beta_m * market + beta_s * sectors[sector_of] + noise
    if n_twins:
        src = rng.choice(n_series - n_twins, size=n_twins, replace=False)
        X[n_series - n_twins:] = X[src] + 0.3 * rng.standard_normal((n_twins, window))
    return 0.01 * X


def zipf_trades(n_trades: int, n_tickers: int = 5000, a: float = 1.2,
                seed: int = 0) -> np.ndarray:
    """Ticker ids for a tape where a few names dominate volume (Zipf)."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_tickers + 1)
    p = ranks ** -a
    p /= p.sum()
    return rng.choice(n_tickers, size=n_trades, p=p)
