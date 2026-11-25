"""Screen a universe for highly correlated pairs using real minute bars.

    python examples/screen_pairs.py --source yahoo           # no key needed
    python examples/screen_pairs.py --source alpaca --start 2026-11-17

Pulls 1-minute bars, builds a (symbols x minutes) return matrix, and runs the
sketch-based correlation screen. Pass --save bars.csv to keep the data for
offline replay later.
"""
import argparse
from collections import defaultdict
from datetime import time
from zoneinfo import ZoneInfo

import numpy as np

from tickstream.feeds import alpaca, replay, yahoo
from tickstream.screen import screen

NY = ZoneInfo("America/New_York")


def regular_hours(bars):
    """Keep 9:30-16:00 ET bars; extended-hours bars are thin and stale."""
    for b in bars:
        t = b.ts.astimezone(NY).time()
        if time(9, 30) <= t < time(16, 0):
            yield b


UNIVERSE = ("AAPL MSFT NVDA AMD AVGO ARM GOOGL GOOG META AMZN TSLA PLTR INTC QCOM MU "
            "TXN ADI NXPI MRVL LRCX AMAT KLAC SPY QQQ IWM DIA XLK SMH SOXX XLF JPM BAC "
            "C WFC GS MS XLE XOM CVX COP KO PEP PG CL V MA PYPL").split()


def to_returns(bars, min_coverage=0.5):
    """Align symbols on one minute grid. IEX bars skip minutes with no trades,
    so forward-fill the last close, drop thinly traded symbols, and drop
    returns that span the overnight gap."""
    closes = defaultdict(dict)
    for b in bars:
        closes[b.symbol][b.ts] = b.close
    grid = sorted({t for c in closes.values() for t in c})
    syms = [s for s in closes if len(closes[s]) >= min_coverage * len(grid)]
    P = np.full((len(syms), len(grid)), np.nan)
    for i, s in enumerate(syms):
        last = np.nan
        for j, t in enumerate(grid):
            last = closes[s].get(t, last)
            P[i, j] = last
    start = int(np.argmax(~np.isnan(P).any(axis=0)))  # first column all symbols traded
    P, grid = P[:, start:], grid[start:]
    R = np.diff(np.log(P), axis=1)
    same_day = np.array([grid[j].date() == grid[j + 1].date() for j in range(len(grid) - 1)])
    return syms, R[:, same_day]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yahoo", "alpaca", "csv"], default="yahoo")
    ap.add_argument("--start", default=None, help="alpaca start date YYYY-MM-DD")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("-k", type=int, default=32)
    ap.add_argument("--minutes", type=int, default=5,
                    help="bar size; 1-min bars understate correlation (Epps effect)")
    a = ap.parse_args()

    if a.source == "yahoo":
        bars = list(yahoo.historical_bars(UNIVERSE, period="5d", interval=f"{a.minutes}m"))
    elif a.source == "alpaca":
        bars = list(alpaca.historical_bars(UNIVERSE, start=a.start or "2026-11-17",
                                           timeframe=f"{a.minutes}Min"))
    else:
        bars = list(replay.replay(a.csv))
    if a.save:
        replay.write_csv(bars, a.save)

    syms, R = to_returns(list(regular_hours(bars)))
    print(f"{len(syms)} symbols x {R.shape[1]} {a.minutes}-minute returns (regular hours)")
    res = screen(R, a.threshold, k=a.k, slack=0.15)
    print(f"{res.n_candidates} candidates from sketches -> {len(res.pairs)} verified pairs\n")
    for i, j, c in res.pairs[:25]:
        print(f"  {syms[i]:6} {syms[j]:6} corr={c:.3f}")


if __name__ == "__main__":
    main()
