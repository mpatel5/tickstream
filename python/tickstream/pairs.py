"""Z-score pairs-trading backtest driven by the C++ PairStats engine.

Research tool only: it ignores slippage beyond a flat cost, borrow costs,
financing and position limits. Not investment advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._core import PairStats


@dataclass
class BacktestResult:
    equity: np.ndarray          # cumulative P&L in log-return units
    position: np.ndarray        # spread position held over (t, t+1]: +1 long, -1 short, 0 flat
    z: np.ndarray               # spread z-score at each tick (no look-ahead)
    beta: np.ndarray            # hedge ratio available at each tick
    trades: int
    total_return: float
    sharpe: float               # per-tick Sharpe, scaled by sqrt(periods_per_year)
    max_drawdown: float
    hit_rate: float             # fraction of closed trades with positive P&L
    trade_pnls: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"trades={self.trades}  return={self.total_return:+.4f}  sharpe={self.sharpe:.2f}  "
            f"max_dd={self.max_drawdown:.4f}  hit_rate={self.hit_rate:.0%}"
        )


def backtest(
    x: np.ndarray,
    y: np.ndarray,
    halflife: float = 500.0,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    warmup: int = 1_000,
    cost_bps: float = 1.0,
    periods_per_year: float = 252 * 390,
) -> BacktestResult:
    """Backtest a mean-reversion strategy on the spread of two log-price series.

    Signals at tick t use only data up to t-1 for the hedge ratio and spread
    statistics (see PairStats). A position opened at t earns the return from t to
    t+1. The hedge ratio is frozen at entry for the life of each trade.
    Short the spread when z > entry_z, buy it when z < -entry_z, exit when
    |z| < exit_z, and stop out when |z| > stop_z.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    n = len(x)

    beta, _, z = PairStats(halflife).update_many(x, y)

    pos = np.zeros(n)        # spread units: +1 = long y / short beta*x
    hedge = np.zeros(n)      # beta frozen at entry
    state, held_beta = 0, 0.0
    for t in range(n):
        if t >= warmup:
            zt = z[t]
            if state == 0:
                if zt > entry_z and zt < stop_z:
                    state, held_beta = -1, beta[t]
                elif zt < -entry_z and zt > -stop_z:
                    state, held_beta = 1, beta[t]
            elif abs(zt) < exit_z or abs(zt) > stop_z:
                state = 0
        pos[t], hedge[t] = state, held_beta if state else 0.0

    dx, dy = np.diff(x), np.diff(y)
    leg_y, leg_x = pos[:-1], -pos[:-1] * hedge[:-1]
    gross = leg_y * dy + leg_x * dx

    # Flat cost on traded notional each time either leg changes.
    turnover = np.abs(np.diff(pos, prepend=0.0)) + np.abs(np.diff(-pos * hedge, prepend=0.0))
    costs = turnover[:-1] * cost_bps / 1e4
    pnl = gross - costs
    equity = np.concatenate([[0.0], np.cumsum(pnl)])

    # Per-trade P&L for hit rate. pnl[t] covers (t, t+1] plus the cost of the
    # position change at t, so a trade's exit cost lands on the first flat tick.
    trade_pnls: list[float] = []
    running = 0.0
    for t in range(n - 1):
        if pos[t] != 0:
            running += pnl[t]
            if t == n - 2 or pos[t + 1] != pos[t]:
                if pos[t + 1] == 0 and t + 1 < n - 1:
                    running += pnl[t + 1]  # exit cost; flat, so no market P&L
                trade_pnls.append(running)
                running = 0.0

    entries = int(np.sum((pos != 0) & (np.concatenate([[0.0], pos[:-1]]) != pos)))
    sd = pnl.std()
    sharpe = float(pnl.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    return BacktestResult(
        equity=equity,
        position=pos,
        z=z,
        beta=beta,
        trades=entries,
        total_return=float(equity[-1]),
        sharpe=sharpe,
        max_drawdown=float(np.max(peak - equity)),
        hit_rate=float(np.mean([p > 0 for p in trade_pnls])) if trade_pnls else 0.0,
        trade_pnls=trade_pnls,
    )
