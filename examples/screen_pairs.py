"""Screen a universe for highly correlated pairs using real intraday bars.

    python examples/screen_pairs.py --source yahoo           # no key needed
    python examples/screen_pairs.py --source alpaca --start 2026-11-17

Pulls 5-minute bars (regular hours), builds a (symbols x bars) return matrix,
and runs the sketch-based correlation screen. Pass --save bars.csv to keep the data for
offline replay later.
"""
import argparse

from tickstream.bars import align, regular_hours
from tickstream.feeds import alpaca, replay, yahoo
from tickstream.screen import screen

UNIVERSE = ("AAPL MSFT NVDA AMD AVGO ARM GOOGL GOOG META AMZN TSLA PLTR INTC QCOM MU "
            "TXN ADI NXPI MRVL LRCX AMAT KLAC SPY QQQ IWM DIA XLK SMH SOXX XLF JPM BAC "
            "C WFC GS MS XLE XOM CVX COP KO PEP PG CL V MA PYPL").split()


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

    panel = align(regular_hours(bars))
    syms, R = panel.symbols, panel.returns()
    print(f"{len(syms)} symbols x {R.shape[1]} {a.minutes}-minute returns (regular hours)")
    res = screen(R, a.threshold, k=a.k, slack=0.15)
    print(f"{res.n_candidates} candidates from sketches -> {len(res.pairs)} verified pairs\n")
    for i, j, c in res.pairs[:25]:
        print(f"  {syms[i]:6} {syms[j]:6} corr={c:.3f}")


if __name__ == "__main__":
    main()
