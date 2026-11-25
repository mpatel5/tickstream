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
| `KLL` (Python) | What are the 1st / 50th / 99th percentiles of trade size or returns? | O(k) |
| `MisraGries` (Python) | Which symbols account for more than φ of all trades? | k counters |
| `screen.screen` (Python) | Which of N symbols have correlation ≥ ρ with each other? | k floats per symbol |

`tickstream.feeds` connects all of this to real market data from Alpaca (live and historical) or Yahoo Finance, and can record and replay it offline.

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

## Real market data

```bash
export ALPACA_API_KEY_ID=...  ALPACA_API_SECRET_KEY=...   # free at alpaca.markets
python examples/screen_pairs.py --source alpaca --start 2026-11-17
python examples/live_monitor.py AAPL MSFT NVDA AMD SPY QQQ  # live, market hours
python examples/screen_pairs.py --source yahoo              # no key needed
```

Every feed emits the same `Trade` / `Bar` events (`tickstream.events`), so a consumer can't tell a live stream from a replay. That's what keeps a backtest honest.

`screen_pairs.py` on 47 large-cap stocks and ETFs, 5-minute bars from Alpaca's IEX feed, Nov 17–24 2026 (462 regular-hours returns). The sketch cut 1,081 possible pairs to 592 candidates, and exact verification kept 238 with ρ ≥ 0.7. The top of the list:

```
GOOG   GOOGL  corr=0.983
SMH    SOXX   corr=0.969
QQQ    XLK    corr=0.966
QQQ    SPY    corr=0.962
MRVL   SOXX   corr=0.923
XLE    XOM    corr=0.887
BAC    WFC    corr=0.884
```

The first run used 1-minute bars including extended hours. GOOG/GOOGL didn't make the list, and BAC/XLF came out at 0.82 instead of 0.88. Two things cause that: thin, stale pre- and post-market prints, and the **Epps effect**, where correlations measured over very short intervals are biased toward zero because two stocks' last trades don't happen at the same instant. The screener now filters to regular hours and defaults to 5-minute bars (`--minutes 1` reproduces the effect).

Correlation is only a first filter. A pair can move together minute to minute and still drift apart over weeks, so candidates go on to a cointegration test.

## Pairs pipeline

```bash
python examples/pair_pipeline.py      # ~3 months of 5-min bars, cached in data/
```

1. **Formation window** (first two-thirds of the bars): correlation screen on returns, then an Engle-Granger cointegration test on log prices (`tickstream.coint`, NumPy only, checked against statsmodels to 6 significant figures). Pairs are ranked by ADF statistic, and each gets a hedge ratio and a mean-reversion half-life.
2. **Trading window** (last third): the existing `pairs.backtest` runs on each selected pair. Its estimators warm up on the formation window but it only trades in the trading window, so the performance comes from data the selection never saw.

The script prints how many pairs would pass by chance. Testing ~200 pairs at 5% produces ~10 false positives, which is why the out-of-sample column matters more than the test statistic.

### Results on real data

47 large-cap stocks and ETFs, 5-minute bars from Alpaca's IEX feed, regular hours. Pairs are chosen on the first two-thirds of each period and traded on the last third.

| period (formation → trading) | correlated | cointegrated at 5% | expected by chance | selected pairs, out of sample |
|---|---:|---:|---:|---|
| Aug 24–Oct 23 → Oct 26–Nov 24, 2026 | 37 | 1 | ~2 | GOOGL/GOOG: 6 trades, +0.46% at 1 bp/leg, −0.02% at 3 bp/leg |
| Nov 24 2025–Jul 29 2026 → Jul 29–Nov 24 2026 | 49 | 2 | ~2 | SMH/SOXX: 13 trades, −0.95%; CVX/XOM: 6 trades, +3.13% |

What this shows:

- **The test passes about as many pairs as chance would.** Over a year, 2 of 49 correlated pairs passed at 5%, the number expected if none were really cointegrated. With this universe, bar size and method, there's no evidence of more than chance, and the out-of-sample split shows it: one selected pair made money and the other lost.
- **Costs decide everything intraday.** GOOGL/GOOG, the one pair with an economic reason to stay linked (two share classes of one company), earned about 12 bp per round trip before costs. A round trip trades both legs twice, so the edge disappears at 3 bp per leg.
- **Even "obvious" pairs drift.** GOOGL/GOOG passed over three months but not over the year, and SMH/SOXX (two semiconductor ETFs with different weightings) passed in formation and then lost money. A hedge ratio fitted on one period doesn't guarantee the next.
- **One week of data overstates correlation.** A one-week screen found 238 pairs with correlation ≥ 0.7. Over three months there were 37.

This matches the academic record: simple distance and cointegration pairs strategies on liquid U.S. stocks became much less profitable after the early 2000s (Do & Faff, *Does Simple Pairs Trading Still Work?*, Financial Analysts Journal, 2010). Next to try: a larger universe with more structural pairs (ETFs on the same index, ADRs vs local shares), rolling re-selection instead of one fixed split, and a cost model based on actual quoted spreads.

## Python benchmarks

`python bench/py_bench.py` writes [`bench/py_results.md`](bench/py_results.md). On synthetic data:

- **KLL:** 500K fat-tailed returns kept in 4.8 KB (vs 3.9 MB exact), max rank error 0.66% at k = 200.
- **Misra-Gries:** 50 counters find every ticker with ≥ 1% of trades.
- **Correlation screen:** 3,000 series × 390 returns, 100% recall of pairs with ρ ≥ 0.8 from a 0.77 MB sketch, vs 9.4 MB of raw windows.

Two caveats:

- **Tail quantiles are KLL's weak spot.** Its error is uniform in rank, so at the 1% tail a small rank error becomes a 2–8% error in VaR. A relative-error sketch (REQ, Cormode et al. 2021) fixes this.
- **The screen isn't much faster than exact on one machine** (~1.2×), because BLAS makes the exact N² product fast. What it saves is memory: k floats per symbol instead of the whole window, which matters as the universe or window grows or the data is spread across machines.

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
pip install -e ".[test,all]"   # all = Alpaca websocket + Yahoo feeds
pytest
python examples/pairs_demo.py
```

Requires a C++20 compiler (GCC 10+ or Clang 12+), CMake 3.16+, and Python 3.10+ with NumPy.

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
python/tickstream/    Python package: pairs backtest, KLL, Misra-Gries, correlation
                      screen, synthetic data
python/tickstream/feeds/  Alpaca, Yahoo and CSV replay feeds
python/tickstream/coint.py, research.py, bars.py
                      cointegration test, pairs pipeline, bar alignment
examples/             pairs demo, pair screener, pairs pipeline, live tape monitor
tests/                C++ unit tests (no dependencies, run with ctest)
python/tests/         pytest suite
bench/                C++ (bench.cpp) and Python (py_bench.py) benchmarks
```

## Roadmap

- [x] Loaders for real historical bar and tick data (Alpaca, Yahoo, CSV replay)
- [x] Correlation pre-screen with random-projection sketches
- [x] Rank screened pairs by cointegration (Engle-Granger) with out-of-sample backtest
- [ ] Paper-trading loop on a live feed
- [ ] Web dashboard for equity curves and spread z-scores
- [x] Streaming quantiles (KLL sketch), in Python
- [ ] Port KLL and Misra-Gries to the C++ core
- [ ] REQ sketch for accurate tail quantiles (VaR)

## License

Copyright (c) 2026 Mitesh Patel. All rights reserved.

The source is public for viewing. No license is granted to use, copy, modify or distribute it without written permission.
