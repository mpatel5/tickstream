"""Pairs research pipeline: screen -> cointegration rank -> out-of-sample backtest.

Pairs are chosen on a formation window and traded only on the window after it,
so the reported performance comes from data the selection never saw.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bars import PricePanel
from .coint import CointResult, engle_granger_both
from .pairs import backtest
from .screen import screen


@dataclass
class PairCandidate:
    a: str
    b: str
    corr: float
    coint: CointResult

    @property
    def y(self) -> str:      # dependent leg in the stronger regression direction
        return self.b if self.coint.swapped else self.a

    @property
    def x(self) -> str:
        return self.a if self.coint.swapped else self.b


@dataclass
class OOSResult:
    pair: PairCandidate
    trades: int
    total_return: float      # sum of log-return P&L per unit of spread, after costs
    sharpe: float            # annualised, trading window only
    max_drawdown: float
    hit_rate: float


def rank_pairs(formation: PricePanel, corr_threshold: float = 0.7, k: int = 32,
               level: float = 0.05, max_pairs: int | None = None) -> tuple[list[PairCandidate], int]:
    """Correlation screen on returns, then Engle-Granger on log prices.

    Returns (pairs passing the cointegration test at `level`, sorted by
    t-statistic, number of correlated candidates tested).
    """
    R = formation.returns()
    hits = screen(R, corr_threshold, k=k, slack=0.15).pairs
    if max_pairs:
        hits = hits[:max_pairs]
    L = formation.log_close
    out = []
    for i, j, c in hits:
        res = engle_granger_both(L[i], L[j])
        if res.significance is not None and res.significance <= level:
            out.append(PairCandidate(formation.symbols[i], formation.symbols[j], c, res))
    out.sort(key=lambda p: p.coint.t_stat)
    return out, len(hits)


def backtest_oos(panel: PricePanel, split: int, pair: PairCandidate, halflife: float,
                 periods_per_year: float, entry_z: float = 2.0, exit_z: float = 0.5,
                 stop_z: float = 4.0, cost_bps: float = 1.0) -> OOSResult:
    """Warm the estimators on bars [0, split), trade only on [split, end)."""
    y = panel.log_close[panel.index(pair.y)]
    x = panel.log_close[panel.index(pair.x)]
    res = backtest(x, y, halflife=halflife, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
                   warmup=split, cost_bps=cost_bps, periods_per_year=periods_per_year)
    pnl = np.diff(res.equity[split:])
    sd = pnl.std()
    eq = np.concatenate([[0.0], np.cumsum(pnl)])
    return OOSResult(
        pair=pair,
        trades=res.trades,
        total_return=float(eq[-1]),
        sharpe=float(pnl.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
        max_drawdown=float(np.max(np.maximum.accumulate(eq) - eq)),
        hit_rate=res.hit_rate,
    )
