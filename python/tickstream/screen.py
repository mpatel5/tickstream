"""Correlation screening with random-projection sketches (StatStream-style).

Z-normalise each window (mean 0, unit norm). For unit vectors,
corr(x, y) = <z_x, z_y> = 1 - ||z_x - z_y||^2 / 2, and a Gaussian random
projection to k dims preserves these distances (Johnson-Lindenstrauss).
So each series is summarised by k floats instead of w, and the all-pairs
comparison costs O(N^2 k) instead of O(N^2 w).

`screen` returns candidate pairs whose *sketched* correlation clears
`threshold - slack`, then verifies them exactly, so the output has no
false positives; recall depends on k and slack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def znormalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Z = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return Z / norms


class CorrelationSketch:
    def __init__(self, window: int, k: int = 32, seed: int = 0):
        self.window, self.k = window, k
        rng = np.random.default_rng(seed)
        self.R = rng.standard_normal((window, k)) / np.sqrt(k)

    def sketch(self, X: np.ndarray) -> np.ndarray:
        """X: (n_series, window) -> (n_series, k) sketch."""
        if X.shape[1] != self.window:
            raise ValueError(f"expected window {self.window}, got {X.shape[1]}")
        return znormalize(X) @ self.R

    @staticmethod
    def approx_corr(S: np.ndarray) -> np.ndarray:
        sq = np.einsum("ij,ij->i", S, S)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (S @ S.T)
        return 1.0 - d2 / 2.0


@dataclass
class ScreenResult:
    pairs: list[tuple[int, int, float]]  # (i, j, exact corr), i < j
    n_candidates: int


def screen(X: np.ndarray, threshold: float, k: int = 32, slack: float = 0.1,
           seed: int = 0) -> ScreenResult:
    cs = CorrelationSketch(X.shape[1], k=k, seed=seed)
    S = cs.sketch(X)
    A = cs.approx_corr(S)
    iu, ju = np.triu_indices(X.shape[0], k=1)
    mask = A[iu, ju] >= threshold - slack
    ci, cj = iu[mask], ju[mask]
    Z = znormalize(X)
    exact = np.einsum("ij,ij->i", Z[ci], Z[cj])
    keep = exact >= threshold
    pairs = [(int(i), int(j), float(c)) for i, j, c in zip(ci[keep], cj[keep], exact[keep])]
    pairs.sort(key=lambda p: -p[2])
    return ScreenResult(pairs=pairs, n_candidates=int(mask.sum()))


def exact_pairs(X: np.ndarray, threshold: float) -> set[tuple[int, int]]:
    Z = znormalize(X)
    C = Z @ Z.T
    iu, ju = np.triu_indices(X.shape[0], k=1)
    m = C[iu, ju] >= threshold
    return set(zip(iu[m].tolist(), ju[m].tolist()))
