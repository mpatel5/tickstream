"""Python-side benchmarks: KLL, Misra-Gries and the correlation screen.

    python bench/py_bench.py      # writes bench/py_results.md

The C++ structures are benchmarked by bench/bench.cpp.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from tickstream import KLL, MisraGries
from tickstream.screen import exact_pairs, screen
from tickstream.synth import factor_returns, zipf_trades

OUT = Path(__file__).with_name("py_results.md")


def bench_kll(n=500_000):
    rng = np.random.default_rng(0)
    data = rng.standard_t(3, size=n) * 0.01  # fat-tailed "returns"
    srt = np.sort(data)
    rows = []
    for k in (50, 100, 200, 400):
        s = KLL(k=k, seed=0)
        t = time.perf_counter()
        s.extend(data.tolist())
        dt = time.perf_counter() - t
        qs = np.linspace(0.001, 0.999, 200)
        errs = [abs(np.searchsorted(srt, s.quantile(q), side="right") / n - q) for q in qs]
        var_err = abs(s.quantile(0.01) - np.quantile(data, 0.01)) / abs(np.quantile(data, 0.01))
        rows.append((k, s.stored_items, s.stored_items * 8 / 1024, max(errs), var_err, n / dt))
    lines = [f"### KLL quantiles — {n:,} Student-t(3) returns (exact = {n * 8 / 1024:,.0f} KB)\n",
             "| k | items kept | memory (KB) | max rank error | 1% VaR rel. error | updates/s |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {m:,} | {kb:.1f} | {e:.4f} | {v:.2%} | {ups:,.0f} |"
              for k, m, kb, e, v, ups in rows]
    return lines


def bench_misra_gries(n=500_000, phi=0.01):
    trades = zipf_trades(n, n_tickers=5000, seed=1)
    true = np.bincount(trades, minlength=5000)
    heavy = set(np.where(true >= phi * n)[0].tolist())
    lines = [f"### Misra-Gries — tickers with >= {phi:.0%} of volume ({len(heavy)} true heavy hitters)\n",
             "| counters | recall | reported | max undercount / N |", "|---|---|---|---|"]
    for k in (50, 100, 200):
        mg = MisraGries(k=k)
        for t in trades.tolist():
            mg.update(t)
        found = {key for key, _ in mg.heavy_hitters(phi)}
        under = max(true[h] - mg.estimate(h) for h in heavy) / n
        lines.append(f"| {k} | {len(heavy & found) / len(heavy):.0%} | {len(found)} | {under:.5f} |")
    return lines


def bench_corr(n_series=3000, window=390, threshold=0.8):
    X = factor_returns(n_series, window, n_sectors=40, seed=0, n_twins=150)
    t = time.perf_counter()
    truth = exact_pairs(X, threshold)
    t_exact = time.perf_counter() - t
    lines = [f"### Correlation screen — {n_series:,} series x {window} minute returns, "
             f"corr >= {threshold} ({len(truth)} true pairs, {n_series * (n_series - 1) // 2:,} total)\n",
             f"Exact all-pairs: {t_exact:.2f}s, comparison data {n_series * window * 8 / 1e6:.1f} MB\n",
             "| sketch dims k | sketch MB | candidates | recall | precision | time (s) | speedup |",
             "|---|---|---|---|---|---|---|"]
    for k in (16, 32, 64):
        t = time.perf_counter()
        res = screen(X, threshold, k=k, slack=0.15)
        dt = time.perf_counter() - t
        found = {(i, j) for i, j, _ in res.pairs}
        rec = len(found & truth) / len(truth)
        prec = len(found & truth) / max(len(found), 1)
        lines.append(f"| {k} | {n_series * k * 8 / 1e6:.2f} | {res.n_candidates:,} | {rec:.1%} | "
                     f"{prec:.0%} | {dt:.2f} | {t_exact / dt:.1f}x |")
    return lines


def main():
    sections = [bench_kll(), bench_misra_gries(), bench_corr()]
    text = "# tickstream Python benchmarks\n\nSynthetic data (`tickstream.synth`); "
    text += f"Python {sys.version.split()[0]}, numpy {np.__version__}.\n\n"
    text += "\n\n".join("\n".join(s) for s in sections) + "\n"
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
