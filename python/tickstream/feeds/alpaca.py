"""Alpaca market data: historical bars (REST) and live trades/bars (WebSocket).

Free account at alpaca.markets gives API keys. Set:
    export ALPACA_API_KEY_ID=...
    export ALPACA_API_SECRET_KEY=...

The free plan streams the IEX feed live; full-market SIP data needs a paid
plan (historical SIP is available with a 15-minute delay).
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from typing import AsyncIterator, Callable, Iterator, Sequence

from ..events import Bar, Event, Trade, parse_ts

REST_URL = "https://data.alpaca.markets/v2/stocks/bars"
WS_URL = "wss://stream.data.alpaca.markets/v2/{feed}"


def _creds(key: str | None, secret: str | None) -> tuple[str, str]:
    key = key or os.environ.get("ALPACA_API_KEY_ID")
    secret = secret or os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY "
                           "(free keys at https://alpaca.markets).")
    return key, secret


# -- message parsing (pure functions, unit-tested) ---------------------------
def parse_bar(sym: str, b: dict) -> Bar:
    return Bar(sym, parse_ts(b["t"]), float(b["o"]), float(b["h"]), float(b["l"]),
               float(b["c"]), float(b["v"]))


def parse_ws_message(msg: dict) -> Event | None:
    t = msg.get("T")
    if t == "t":
        return Trade(msg["S"], parse_ts(msg["t"]), float(msg["p"]), float(msg["s"]))
    if t == "b":
        return parse_bar(msg["S"], msg)
    if t == "error":
        raise RuntimeError(f"Alpaca stream error {msg.get('code')}: {msg.get('msg')}")
    return None  # success / subscription acks, statuses, etc.


# -- historical ---------------------------------------------------------------
def _http_get_json(url: str, headers: dict, retries: int = 5) -> dict:
    """GET with retries on rate limits (429), server errors and timeouts."""
    import time
    import urllib.error

    delay = 1.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
            wait = float(e.headers.get("Retry-After") or delay)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
            wait = delay
        time.sleep(wait)
        delay = min(delay * 2, 30.0)
    raise RuntimeError("unreachable")


def historical_bars(symbols: Sequence[str], start: str | datetime,
                    end: str | datetime | None = None, timeframe: str = "1Min",
                    feed: str = "iex", adjustment: str = "all",
                    key: str | None = None, secret: str | None = None,
                    fetch: Callable[[str, dict], dict] = _http_get_json,
                    progress: Callable[[int, int], None] | None = None) -> Iterator[Bar]:
    """Yield bars for all symbols in timestamp order (handles pagination).

    `progress(pages, bars_so_far)` is called after each page, if given.
    """
    key, secret = _creds(key, secret)
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {"symbols": ",".join(symbols), "timeframe": timeframe, "feed": feed,
              "adjustment": adjustment, "limit": 10000,
              "start": start.isoformat() if isinstance(start, datetime) else start}
    if end is not None:
        params["end"] = end.isoformat() if isinstance(end, datetime) else end
    bars: list[Bar] = []
    token, pages = None, 0
    while True:
        q = dict(params, **({"page_token": token} if token else {}))
        data = fetch(f"{REST_URL}?{urllib.parse.urlencode(q)}", headers)
        for sym, rows in (data.get("bars") or {}).items():
            bars.extend(parse_bar(sym, b) for b in rows)
        pages += 1
        if progress:
            progress(pages, len(bars))
        token = data.get("next_page_token")
        if not token:
            break
    # API sorts by symbol then time; re-sort so consumers see one timeline.
    bars.sort(key=lambda b: (b.ts, b.symbol))
    yield from bars


# -- live -----------------------------------------------------------------------
async def stream(symbols: Sequence[str], trades: bool = True, bars: bool = False,
                 feed: str = "iex", key: str | None = None, secret: str | None = None,
                 reconnect: bool = True) -> AsyncIterator[Event]:
    """Async iterator of live Trade/Bar events. Reconnects with backoff."""
    import websockets  # optional dependency: pip install tickstream[live]

    key, secret = _creds(key, secret)
    sub = {"action": "subscribe"}
    if trades:
        sub["trades"] = list(symbols)
    if bars:
        sub["bars"] = list(symbols)
    delay = 1.0
    while True:
        try:
            async with websockets.connect(WS_URL.format(feed=feed)) as ws:
                await ws.recv()  # [{"T":"success","msg":"connected"}]
                await ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))
                for m in json.loads(await ws.recv()):
                    parse_ws_message(m)  # raises on auth error
                await ws.send(json.dumps(sub))
                delay = 1.0
                async for raw in ws:
                    for m in json.loads(raw):
                        ev = parse_ws_message(m)
                        if ev is not None:
                            yield ev
            if not reconnect:
                return
        except (OSError, websockets.ConnectionClosed):
            if not reconnect:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
