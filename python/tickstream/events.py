"""The one event format every feed emits and every consumer reads.

Live and historical sources produce the same objects, so a strategy or
sketch can't tell (and can't depend on) whether it's being backtested.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    ts: datetime          # tz-aware UTC
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    ts: datetime          # bar open time, tz-aware UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


Event = Trade | Bar


def parse_ts(s: str) -> datetime:
    """RFC-3339 (with optional nanoseconds / 'Z') -> aware UTC datetime."""
    s = s.strip().replace("Z", "+00:00")
    if "." in s:
        head, rest = s.split(".", 1)
        frac = "".join(ch for ch in rest if ch.isdigit())
        tz = rest[len(frac):]
        s = f"{head}.{frac[:6].ljust(6, '0')}{tz}"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
