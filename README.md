# tickstream

Streaming sketch algorithms and pair statistics for market data, written as a header-only C++20 core with Python bindings (pybind11).

Market data arrives as an unbounded stream. tickstream answers common questions about that stream in a single pass, with memory that stays fixed no matter how much data comes through:

| Structure | Question it answers | Memory |
|---|---|---|
| `CountMinSketch` | How much volume has traded in symbol / price level *x*? | width × depth counters |
| `HyperLogLog` | How many distinct order IDs, symbols or counterparties have we seen? | 2^p bytes |
| `Reservoir` (Algorithm L) | Give me a uniform random sample of *k* ticks from the stream | *k* items |
| `RunningStats` / `EwmStats` | Mean, variance and z-score of a price or spread, cumulative or decaying | O(1) |
| `PairStats` | Rolling hedge ratio, spread and spread z-score for a pair of instruments | O(1) |

On top of the core, `tickstream.pairs` runs a z-score mean-reversion backtest for a pair of instruments, using `PairStats` to generate the signals.

## Benchmark

`./build/bench` processes 10M ticks drawn from a skewed 1M-symbol universe. The figures below are from one run on a shared cloud Linux VM built with `-O3 -march=native`, so expect yours to differ:

| Structure | M updates/s | Memory |
|---|---:|---:|
| Count-Min (ε = 1e-4, δ = 1e-3) | 31.4 | 1.52 MB |
| Count-Min, conservative update | 8.3 | 1.52 MB |
| `std::unordered_map` (exact counts) | 18.8 | ~43 MB |
| HyperLogLog (p = 14) | 207.1 | 16 KB |
| Reservoir, Algorithm L (k = 1024) | 390.3 | 8 KB |
| PairStats (EW beta + z-score) | 121.9 | < 1 KB |

Over ~990K distinct keys, Count-Min overcounted by 215 on average, with a maximum of 484. Its guaranteed bound is εN = 1,000. Conservative update cut the average overcount roughly in half, to 105, in exchange for slower updates. HyperLogLog's error is typically under 1% at p = 14, which has a standard error of 0.81%.

## Design notes

- **One-sided error.** Count-Min never undercounts: `true(x) ≤ estimate(x) ≤ true(x) + εN` with probability ≥ 1 − δ. `from_error(ε, δ)` sizes the table as w = ⌈e/ε⌉ and d = ⌈ln(1/δ)⌉.
- **Mergeable.** Count-Min and HyperLogLog sketches built with the same shape and seed can be merged, so each shard or feed handler can keep its own sketch and the results can be combined afterward.
- **Hashing.** Every row uses a seeded SplitMix64 mixer. Lemire's fast-range maps hashes onto table buckets without a modulo, and string keys such as tickers go through FNV-1a first.
- **Algorithm L.** Instead of drawing a random number for every item, the reservoir draws how many items to skip. The expected number of draws is O(k(1 + log(n/k))) rather than O(n), which is why it's the fastest structure in the table above.
- **No look-ahead.** `PairStats::update` scores each tick using the hedge ratio and spread statistics estimated from *earlier* ticks, and only then folds the tick in. The backtest also freezes the hedge ratio at entry for the life of each trade.

## Build

```bash
# C++ tests and benchmark
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench

# Python package
pip install -e ".[test]"
pytest
python examples/pairs_demo.py
```

Requires a C++20 compiler (GCC 10+ or Clang 12+), CMake 3.16+, and Python 3.9+ with NumPy.

## Python usage

```python
import numpy as np
import tickstream as ts
from tickstream.pairs import backtest
from tickstream.synth import cointegrated_pair

volume = ts.CountMinSketch.from_error(epsilon=1e-4, delta=1e-3)
volume.add("NVDA", 300)
volume["NVDA"]            # >= 300

distinct = ts.HyperLogLog(precision=14)
distinct.add_many(np.arange(1_000_000, dtype=np.uint64))
distinct.estimate()       # ~1,000,000

x, y = cointegrated_pair(n=50_000)          # log prices
beta, spread, z = ts.PairStats(halflife=1_000).update_many(x, y)
print(backtest(x, y, halflife=1_000).summary())
```

Here is the output of `examples/pairs_demo.py`:

```
cointegrated pair : trades=170  return=+4.8652  sharpe=19.24  max_dd=0.0556  hit_rate=99%
independent walks : trades=29  return=+0.1139  sharpe=0.14  max_dd=1.2489  hit_rate=62%
```

These are **synthetic** results that check the engine behaves sensibly. It finds an edge where a mean-reverting relationship was planted, and essentially none between two unrelated random walks. They say nothing about real markets. The cointegrated series is idealized, and the backtest models costs as a flat bps charge, with no slippage, borrow costs or position limits.

## Project layout

```
include/tickstream/   header-only C++ core
src/bindings.cpp      pybind11 module (tickstream._core)
python/tickstream/    Python package: pairs backtest, synthetic data
tests/                C++ unit tests (no dependencies, run with ctest)
python/tests/         pytest suite
bench/                throughput and accuracy benchmark
```

## Roadmap

- [ ] Loaders for real historical bar and tick data
- [ ] Screener that ranks candidate pairs by cointegration test statistics
- [ ] Paper-trading loop on a live feed
- [ ] Web dashboard for equity curves and spread z-scores
- [ ] Streaming quantiles (KLL sketch) for spread and latency distributions

## License

Copyright (c) 2026 Mitesh Patel. All rights reserved.

The source is public for viewing. No license is granted to use, copy, modify or distribute it without written permission.
