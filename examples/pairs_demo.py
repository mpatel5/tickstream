"""Run the pairs backtest on synthetic data and print the results.

    python examples/pairs_demo.py
"""

import numpy as np

from tickstream import CountMinSketch, HyperLogLog
from tickstream.pairs import backtest
from tickstream.synth import cointegrated_pair, independent_pair

x, y = cointegrated_pair(n=50_000, beta=1.3, seed=7)
print("cointegrated pair :", backtest(x, y, halflife=1_000, warmup=2_000).summary())

x, y = independent_pair(n=50_000, seed=7)
print("independent walks :", backtest(x, y, halflife=1_000, warmup=2_000).summary())

# Sketches on a simulated tape: volume by symbol and distinct order IDs.
rng = np.random.default_rng(0)
symbols = np.array(["AAPL", "MSFT", "NVDA", "ARM", "AVGO"])
tape_syms = rng.choice(symbols, size=200_000, p=[0.35, 0.25, 0.2, 0.1, 0.1])
tape_vols = rng.integers(1, 500, size=200_000)

volume = CountMinSketch.from_error(1e-4, 1e-3)
orders = HyperLogLog(14)
for sym, vol in zip(tape_syms, tape_vols):
    volume.add(str(sym), int(vol))
orders.add_many(rng.integers(0, 2**63, size=200_000, dtype=np.uint64))

for sym in symbols:
    exact = int(tape_vols[tape_syms == sym].sum())
    print(f"{sym:5s} volume  sketch={volume[sym]:>10,}  exact={exact:>10,}")
print(f"distinct order IDs ~ {orders.estimate():,.0f} (true 200,000, {orders.memory_bytes:,} bytes)")
