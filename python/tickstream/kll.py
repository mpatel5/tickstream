"""KLL quantile sketch (Karnin, Lang, Liberty 2016).

Keeps a stack of "compactors". Level h holds items each standing for 2**h
original items. When a level fills, it is sorted and every other item
(random offset) is promoted to the next level. Space is O(k) for
rank error ~ O(1/k), independent of stream length.
"""
from __future__ import annotations

import math
import random
from bisect import bisect_right
from typing import Iterable


class KLL:
    def __init__(self, k: int = 200, c: float = 2.0 / 3.0, seed: int | None = None):
        if k < 8:
            raise ValueError("k must be >= 8")
        self.k = k
        self.c = c
        self.n = 0
        self._rng = random.Random(seed)
        self.compactors: list[list[float]] = []
        self._size = 0
        self._max_size = 0
        self._grow()

    # -- structure ---------------------------------------------------------
    def _capacity(self, h: int) -> int:
        depth = len(self.compactors) - h - 1
        return int(math.ceil(self.k * self.c**depth)) + 1

    def _grow(self) -> None:
        self.compactors.append([])
        self._max_size = sum(self._capacity(h) for h in range(len(self.compactors)))

    def _compact_level(self, h: int) -> None:
        buf = self.compactors[h]
        buf.sort()
        keep_last = buf.pop() if len(buf) % 2 == 1 else None
        offset = self._rng.getrandbits(1)
        promoted = buf[offset::2]
        self.compactors[h] = [keep_last] if keep_last is not None else []
        if h + 1 >= len(self.compactors):
            self._grow()
        self.compactors[h + 1].extend(promoted)

    def _compress(self) -> None:
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self._capacity(h):
                self._compact_level(h)
                self._size = sum(len(c) for c in self.compactors)
                if self._size < self._max_size:
                    break

    # -- public API ----------------------------------------------------------
    def update(self, x: float) -> None:
        self.compactors[0].append(float(x))
        self.n += 1
        self._size += 1
        if self._size >= self._max_size:
            self._compress()

    def extend(self, xs: Iterable[float]) -> None:
        for x in xs:
            self.update(x)

    def merge(self, other: "KLL") -> None:
        while len(self.compactors) < len(other.compactors):
            self._grow()
        for h, items in enumerate(other.compactors):
            self.compactors[h].extend(items)
        self.n += other.n
        self._size = sum(len(c) for c in self.compactors)
        while self._size >= self._max_size:
            self._compress()

    def _weighted(self) -> tuple[list[float], list[int]]:
        pairs = sorted(
            (x, 1 << h) for h, items in enumerate(self.compactors) for x in items
        )
        values = [p[0] for p in pairs]
        cum, total = [], 0
        for _, w in pairs:
            total += w
            cum.append(total)
        return values, cum

    def rank(self, x: float) -> float:
        """Approximate fraction of items <= x."""
        values, cum = self._weighted()
        if not values:
            return 0.0
        i = bisect_right(values, x)
        return (cum[i - 1] / cum[-1]) if i else 0.0

    def quantile(self, q: float) -> float:
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1]")
        values, cum = self._weighted()
        if not values:
            raise ValueError("empty sketch")
        target = q * cum[-1]
        i = bisect_right(cum, target)
        return values[min(i, len(values) - 1)]

    @property
    def stored_items(self) -> int:
        return sum(len(c) for c in self.compactors)

    def __len__(self) -> int:
        return self.n
