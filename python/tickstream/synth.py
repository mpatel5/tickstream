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
