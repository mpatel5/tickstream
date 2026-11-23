"""tickstream: streaming sketches and pair statistics for market data.

The data structures are implemented in C++ (tickstream._core); the pairs-trading
research tools in tickstream.pairs and tickstream.synth are pure Python + NumPy.
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

__version__ = "0.1.0"

__all__ = [
    "CountMinSketch",
    "EwmStats",
    "HyperLogLog",
    "PairStats",
    "Reservoir",
    "RunningStats",
    "hash_key",
]
