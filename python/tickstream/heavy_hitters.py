"""Misra-Gries heavy hitters.

With k counters, every item with frequency > N/(k+1) is guaranteed to be
kept, and each count is underestimated by at most N/(k+1).
"""
from __future__ import annotations


class MisraGries:
    def __init__(self, k: int = 100):
        self.k = k
        self.counters: dict = {}
        self.n = 0

    def update(self, key, count: int = 1) -> None:
        self.n += count
        if key in self.counters or len(self.counters) < self.k:
            self.counters[key] = self.counters.get(key, 0) + count
            return
        dec = min(count, min(self.counters.values()))
        count -= dec
        for kk in list(self.counters):
            self.counters[kk] -= dec
            if self.counters[kk] <= 0:
                del self.counters[kk]
        if count > 0 and len(self.counters) < self.k:
            self.counters[key] = count

    def estimate(self, key) -> int:
        return self.counters.get(key, 0)

    def heavy_hitters(self, phi: float) -> list[tuple]:
        """Candidates with frequency possibly >= phi * N (no false negatives)."""
        thresh = phi * self.n - self.n / (self.k + 1)
        return sorted(((k, v) for k, v in self.counters.items() if v >= thresh),
                      key=lambda kv: -kv[1])

    @property
    def error_bound(self) -> float:
        return self.n / (self.k + 1)
