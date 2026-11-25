"""Live tape monitor: stream trades from Alpaca into sketches.

Prints, every N seconds, per-symbol trade-size quantiles (KLL), the most
active symbols (Misra-Gries) and EW volatility of trade-to-trade returns.

    export ALPACA_API_KEY_ID=... ALPACA_API_SECRET_KEY=...
    python examples/live_monitor.py AAPL MSFT NVDA AMD SPY QQQ

Runs only during market hours (IEX feed is quiet otherwise).
"""
import asyncio
import math
import sys
import time
from collections import defaultdict

from tickstream.feeds import alpaca
from tickstream import KLL, EwmStats, MisraGries

REPORT_EVERY = 10.0


async def main(symbols):
    sizes = defaultdict(lambda: KLL(k=128))
    vol = defaultdict(lambda: EwmStats(halflife=200))
    last_px = {}
    active = MisraGries(k=20)
    n, t0 = 0, time.monotonic()

    async for tr in alpaca.stream(symbols, trades=True):
        n += 1
        sizes[tr.symbol].update(tr.size)
        active.update(tr.symbol)
        if tr.symbol in last_px and last_px[tr.symbol] > 0:
            vol[tr.symbol].add(math.log(tr.price / last_px[tr.symbol]))
        last_px[tr.symbol] = tr.price

        if time.monotonic() - t0 >= REPORT_EVERY:
            t0 = time.monotonic()
            print(f"\n{n:,} trades | most active: "
                  + ", ".join(f"{s}({c})" for s, c in active.heavy_hitters(0.05)))
            for s in sorted(sizes):
                k = sizes[s]
                print(f"  {s:6} px={last_px[s]:>9.2f}  size p50={k.quantile(.5):>6.0f} "
                      f"p99={k.quantile(.99):>7.0f}  tick-vol={vol[s].stddev * 1e4:5.2f}bp")


if __name__ == "__main__":
    syms = sys.argv[1:] or ["AAPL", "MSFT", "NVDA", "AMD", "SPY", "QQQ"]
    try:
        asyncio.run(main(syms))
    except KeyboardInterrupt:
        pass
