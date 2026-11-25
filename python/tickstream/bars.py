"""Turn a stream of Bar events into aligned price and return matrices."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time
from typing import Iterable, Iterator
from zoneinfo import ZoneInfo

import numpy as np

from .events import Bar

NY = ZoneInfo("America/New_York")


def regular_hours(bars: Iterable[Bar]) -> Iterator[Bar]:
    """Keep 9:30-16:00 ET bars; extended-hours bars are thin and stale."""
    for b in bars:
        t = b.ts.astimezone(NY).time()
        if time(9, 30) <= t < time(16, 0):
            yield b


@dataclass
class PricePanel:
    symbols: list[str]
    times: list[datetime]
    close: np.ndarray                # (n_symbols, n_times), forward-filled

    @property
    def log_close(self) -> np.ndarray:
        return np.log(self.close)

    def returns(self, drop_overnight: bool = True) -> np.ndarray:
        """Log returns; by default drops the close-to-open jump between sessions."""
        R = np.diff(self.log_close, axis=1)
        if not drop_overnight:
            return R
        same_day = np.array([self.times[j].astimezone(NY).date() == self.times[j + 1].astimezone(NY).date()
                             for j in range(len(self.times) - 1)], dtype=bool)
        return R[:, same_day]

    def slice(self, start: int, stop: int | None = None) -> "PricePanel":
        return PricePanel(self.symbols, self.times[start:stop], self.close[:, start:stop])

    def index(self, symbol: str) -> int:
        return self.symbols.index(symbol)


def align(bars: Iterable[Bar], min_coverage: float = 0.5) -> PricePanel:
    """Put every symbol on one shared time grid.

    Free feeds (IEX) skip bars with no trades, so the last close is carried
    forward. Symbols present in fewer than `min_coverage` of the grid's bars
    are dropped, and the grid starts at the first bar where every remaining
    symbol has a price.
    """
    closes: dict[str, dict[datetime, float]] = defaultdict(dict)
    for b in bars:
        closes[b.symbol][b.ts] = b.close
    grid = sorted({t for c in closes.values() for t in c})
    syms = sorted(s for s in closes if len(closes[s]) >= min_coverage * len(grid))
    P = np.full((len(syms), len(grid)), np.nan)
    for i, s in enumerate(syms):
        last = np.nan
        c = closes[s]
        for j, t in enumerate(grid):
            last = c.get(t, last)
            P[i, j] = last
    start = int(np.argmax(~np.isnan(P).any(axis=0))) if len(grid) else 0
    return PricePanel(syms, grid[start:], P[:, start:])
