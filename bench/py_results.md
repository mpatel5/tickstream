# tickstream Python benchmarks

Synthetic data (`tickstream.synth`); Python 3.13.18, numpy 2.5.3.

### KLL quantiles — 500,000 Student-t(3) returns (exact = 3,906 KB)

| k | items kept | memory (KB) | max rank error | 1% VaR rel. error | updates/s |
|---|---|---|---|---|---|
| 50 | 168 | 1.3 | 0.0184 | 21.54% | 862,048 |
| 100 | 314 | 2.5 | 0.0167 | 3.16% | 1,485,833 |
| 200 | 609 | 4.8 | 0.0066 | 7.89% | 2,485,596 |
| 400 | 1,186 | 9.3 | 0.0023 | 2.43% | 3,510,454 |

### Misra-Gries — tickers with >= 1% of volume (12 true heavy hitters)

| counters | recall | reported | max undercount / N |
|---|---|---|---|
| 50 | 100% | 35 | 0.01150 |
| 100 | 100% | 23 | 0.00478 |
| 200 | 100% | 17 | 0.00194 |

### Correlation screen — 3,000 series x 390 minute returns, corr >= 0.8 (150 true pairs, 4,498,500 total)

Exact all-pairs: 0.16s, comparison data 9.4 MB

| sketch dims k | sketch MB | candidates | recall | precision | time (s) | speedup |
|---|---|---|---|---|---|---|
| 16 | 0.38 | 92,059 | 100.0% | 100% | 0.43 | 0.4x |
| 32 | 0.77 | 6,838 | 100.0% | 100% | 0.13 | 1.2x |
| 64 | 1.54 | 543 | 100.0% | 100% | 0.13 | 1.3x |
