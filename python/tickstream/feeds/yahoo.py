"""Yahoo Finance historical bars via yfinance (no API key needed).

Good for quick experiments. Limits: 1-minute bars only cover about the last
7 days (~30 days total in 7-day chunks); daily bars go back decades. It's an
unofficial source, so don't build anything that needs to be reliable on it.
"""
from __future__ import annotations

from datetime import timezone
from typing import Iterator, Sequence

from ..events import Bar


def historical_bars(symbols: Sequence[str], period: str = "5d",
                    interval: str = "1m") -> Iterator[Bar]:
    import yfinance as yf  # optional dependency: pip install tickstream[yahoo]

    df = yf.download(list(symbols), period=period, interval=interval,
                     group_by="ticker", auto_adjust=True, progress=False, threads=True)
    bars: list[Bar] = []
    for sym in symbols:
        sub = df[sym] if len(symbols) > 1 or sym in df.columns.get_level_values(0) else df
        sub = sub.dropna(subset=["Close"])
        for ts, row in sub.iterrows():
            ts = ts.to_pydatetime()
            ts = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
            bars.append(Bar(sym, ts, float(row["Open"]), float(row["High"]),
                            float(row["Low"]), float(row["Close"]), float(row["Volume"])))
    bars.sort(key=lambda b: (b.ts, b.symbol))
    yield from bars
