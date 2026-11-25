"""tickstream: streaming sketches and pair statistics for market data.

The core data structures are implemented in C++ (tickstream._core). KLL and
Misra-Gries, the correlation screener (tickstream.screen), market data feeds
(tickstream.feeds) and the pairs research tools are pure Python + NumPy.
"""

from ._core import (
    CountMinSketch,
    EwmStats,
    HyperLogLog,
    PairStats,
    Reservoir,
    RunningStats,
    hash_key,
)
from .heavy_hitters import MisraGries
from .kll import KLL

__version__ = "0.1.0"

__all__ = [
    "CountMinSketch",
    "EwmStats",
    "HyperLogLog",
    "PairStats",
    "Reservoir",
    "RunningStats",
    "hash_key",
    "KLL",
    "MisraGries",
]
