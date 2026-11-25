"""Offline replay: CSV files -> time-ordered event stream.

CSV columns:
  bars:   symbol,ts,open,high,low,close,volume
  trades: symbol,ts,price,size
"""
from __future__ import annotations

import csv
import heapq
from pathlib import Path
from typing import Iterable, Iterator

from ..events import Bar, Event, Trade, parse_ts


def read_csv(path: str | Path) -> Iterator[Event]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        is_bar = {"open", "close"} <= cols
        for row in r:
            ts = parse_ts(row["ts"])
            if is_bar:
                yield Bar(row["symbol"], ts, float(row["open"]), float(row["high"]),
                          float(row["low"]), float(row["close"]), float(row["volume"]))
            else:
                yield Trade(row["symbol"], ts, float(row["price"]), float(row["size"]))


def merge_streams(*streams: Iterable[Event]) -> Iterator[Event]:
    """Merge already time-sorted streams into one time-sorted stream."""
    return heapq.merge(*streams, key=lambda e: e.ts)


def replay(*paths: str | Path) -> Iterator[Event]:
    """Replay one or more CSVs in timestamp order (each file sorted by ts)."""
    return merge_streams(*(sorted(read_csv(p), key=lambda e: e.ts) for p in paths))


def write_csv(events: Iterable[Event], path: str | Path) -> int:
    events = list(events)
    if not events:
        return 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if isinstance(events[0], Bar):
            w.writerow(["symbol", "ts", "open", "high", "low", "close", "volume"])
            for e in events:
                w.writerow([e.symbol, e.ts.isoformat(), e.open, e.high, e.low, e.close, e.volume])
        else:
            w.writerow(["symbol", "ts", "price", "size"])
            for e in events:
                w.writerow([e.symbol, e.ts.isoformat(), e.price, e.size])
    return len(events)
