"""Alpaca *paper* trading account: clock, account, positions, orders.

This module only ever talks to paper-api.alpaca.markets. There is no switch
for live trading on purpose: paper money only.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .events import parse_ts
from .feeds.alpaca import _creds

PAPER_URL = "https://paper-api.alpaca.markets"

Http = Callable[[str, str, dict, dict | None], object]


def _http(method: str, url: str, headers: dict, body: dict | None = None, retries: int = 3):
    data = json.dumps(body).encode() if body is not None else None
    delay = 1.0
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={**headers, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"Alpaca {method} {url} -> {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2


@dataclass
class Clock:
    now: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


class PaperBroker:
    def __init__(self, key: str | None = None, secret: str | None = None, http: Http = _http):
        key, secret = _creds(key, secret)
        self._h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._http = http

    def _call(self, method: str, path: str, body: dict | None = None):
        return self._http(method, PAPER_URL + path, self._h, body)

    def clock(self) -> Clock:
        c = self._call("GET", "/v2/clock")
        return Clock(parse_ts(c["timestamp"]), bool(c["is_open"]),
                     parse_ts(c["next_open"]), parse_ts(c["next_close"]))

    def equity(self) -> float:
        return float(self._call("GET", "/v2/account")["equity"])

    def positions(self) -> dict[str, int]:
        """Signed share counts: negative = short."""
        return {p["symbol"]: int(float(p["qty"])) for p in self._call("GET", "/v2/positions") or []}

    def market_order(self, symbol: str, qty: int) -> dict:
        """Signed qty: positive buys, negative sells (or shorts)."""
        if qty == 0:
            raise ValueError("qty must be non-zero")
        return self._call("POST", "/v2/orders", {
            "symbol": symbol, "qty": str(abs(qty)), "side": "buy" if qty > 0 else "sell",
            "type": "market", "time_in_force": "day",
        })
