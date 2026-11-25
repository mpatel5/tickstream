"""Full pairs pipeline on real data: screen -> cointegration -> out-of-sample backtest.

    python examples/pair_pipeline.py                         # Alpaca, ~3 months of 5-min bars
    python examples/pair_pipeline.py --start 2026-06-01 --top 15
    python examples/pair_pipeline.py --csv data/bars.csv     # replay a saved download

The first run downloads bars and caches them in data/ so later runs are instant.
Pairs are selected on the first --formation fraction of the data and traded
only on the rest.
"""
import argparse
from datetime import date
from pathlib import Path

from tickstream.bars import align, regular_hours
from tickstream.feeds import alpaca, replay
from tickstream.research import backtest_oos, rank_pairs

UNIVERSE = ("AAPL MSFT NVDA AMD AVGO ARM GOOGL GOOG META AMZN TSLA PLTR INTC QCOM MU "
            "TXN ADI NXPI MRVL LRCX AMAT KLAC SPY QQQ IWM DIA XLK SMH SOXX XLF JPM BAC "
            "C WFC GS MS XLE XOM CVX COP KO PEP PG CL V MA PYPL").split()


def load_bars(a):
    if a.csv:
        return list(replay.replay(a.csv))
    cache = Path("data") / f"alpaca_{a.minutes}min_{a.start}_{a.end}.csv"
    if cache.exists():
        print(f"using cached {cache}")
        return list(replay.replay(cache))
    print(f"downloading {len(UNIVERSE)} symbols, {a.minutes}-min bars, {a.start} to {a.end} ...")
    bars = list(alpaca.historical_bars(UNIVERSE, start=a.start, end=a.end,
                                       timeframe=f"{a.minutes}Min"))
    cache.parent.mkdir(exist_ok=True)
    replay.write_csv(bars, cache)
    print(f"cached {len(bars):,} bars -> {cache}")
    return bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-24")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--csv", default=None)
    ap.add_argument("--minutes", type=int, default=5)
    ap.add_argument("--formation", type=float, default=2 / 3,
                    help="fraction of bars used to pick pairs")
    ap.add_argument("--corr", type=float, default=0.7, help="correlation screen threshold")
    ap.add_argument("--level", type=float, default=0.05, choices=[0.01, 0.05, 0.10])
    ap.add_argument("--halflife", type=float, default=390,
                    help="PairStats halflife in bars (390 x 5-min = 5 trading days)")
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    panel = align(regular_hours(load_bars(a)))
    split = int(len(panel.times) * a.formation)
    form = panel.slice(0, split)
    print(f"\n{len(panel.symbols)} symbols, {len(panel.times):,} bars "
          f"({panel.times[0]:%Y-%m-%d} to {panel.times[-1]:%Y-%m-%d})")
    print(f"formation: {form.times[0]:%Y-%m-%d} to {form.times[-1]:%Y-%m-%d} | "
          f"trading: {panel.times[split]:%Y-%m-%d} to {panel.times[-1]:%Y-%m-%d}\n")

    pairs, n_tested = rank_pairs(form, corr_threshold=a.corr, level=a.level)
    print(f"{n_tested} pairs with corr >= {a.corr}; {len(pairs)} cointegrated at {a.level:.0%} "
          f"(~{n_tested * a.level:.0f} would pass by chance alone)\n")
    if not pairs:
        return

    ppy = 252 * 390 / a.minutes
    hours = a.minutes / 60
    print(f"{'pair':14} {'corr':>5} {'EG t':>6} {'crit':>6} {'half-life':>9} {'beta':>5} | "
          f"{'trades':>6} {'return':>7} {'sharpe':>6} {'max dd':>6} {'hit':>4}")
    print("-" * 92)
    total = 0.0
    for p in pairs[: a.top]:
        r = backtest_oos(panel, split, p, halflife=a.halflife, periods_per_year=ppy,
                         cost_bps=a.cost_bps)
        total += r.total_return
        hl = f"{p.coint.half_life * hours:.1f}h"
        print(f"{p.y + '/' + p.x:14} {p.corr:5.2f} {p.coint.t_stat:6.2f} "
              f"{p.coint.crit[a.level]:6.2f} {hl:>9} {p.coint.beta:5.2f} | "
              f"{r.trades:6d} {r.total_return:+7.2%} {r.sharpe:6.2f} {r.max_drawdown:6.2%} "
              f"{r.hit_rate:4.0%}")
    print("-" * 92)
    print(f"equal-weight average out-of-sample return: {total / min(a.top, len(pairs)):+.2%} "
          f"(after {a.cost_bps:g} bp per leg per trade)")


if __name__ == "__main__":
    main()
