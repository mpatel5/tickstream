import json
from datetime import datetime, timezone

import pytest

from tickstream.events import Bar, Trade, parse_ts
from tickstream.feeds import alpaca, replay


def test_parse_ts_nanoseconds():
    dt = parse_ts("2026-11-24T14:30:00.123456789Z")
    assert dt == datetime(2026, 11, 24, 14, 30, 0, 123456, tzinfo=timezone.utc)


def test_parse_ws_trade_and_bar():
    t = alpaca.parse_ws_message({"T": "t", "S": "AAPL", "i": 1, "x": "V", "p": 231.5,
                                 "s": 100, "c": ["@"], "t": "2026-11-24T14:30:01.5Z", "z": "C"})
    assert t == Trade("AAPL", parse_ts("2026-11-24T14:30:01.5Z"), 231.5, 100.0)
    b = alpaca.parse_ws_message({"T": "b", "S": "SPY", "o": 1, "h": 2, "l": 0.5, "c": 1.5,
                                 "v": 1000, "vw": 1.2, "n": 9, "t": "2026-11-24T14:31:00Z"})
    assert isinstance(b, Bar) and b.close == 1.5
    assert alpaca.parse_ws_message({"T": "success", "msg": "authenticated"}) is None
    with pytest.raises(RuntimeError):
        alpaca.parse_ws_message({"T": "error", "code": 402, "msg": "auth failed"})


def test_historical_bars_paginates_and_time_sorts():
    pages = [
        {"bars": {"AAPL": [{"t": "2026-11-24T14:30:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1},
                           {"t": "2026-11-24T14:31:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 2}]},
         "next_page_token": "abc"},
        {"bars": {"MSFT": [{"t": "2026-11-24T14:30:00Z", "o": 3, "h": 3, "l": 3, "c": 3, "v": 3}]},
         "next_page_token": None},
    ]
    seen_urls = []

    def fake_fetch(url, headers):
        seen_urls.append(url)
        assert headers["APCA-API-KEY-ID"] == "k"
        return pages[len(seen_urls) - 1]

    bars = list(alpaca.historical_bars(["AAPL", "MSFT"], start="2026-11-24",
                                       key="k", secret="s", fetch=fake_fetch))
    assert [(b.symbol, b.close) for b in bars] == [("AAPL", 1), ("MSFT", 3), ("AAPL", 2)]
    assert "page_token=abc" in seen_urls[1]


def test_missing_creds_message(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPACA_API_KEY_ID"):
        list(alpaca.historical_bars(["AAPL"], start="2026-11-24"))


def test_csv_roundtrip_and_merge(tmp_path):
    ts = [parse_ts(f"2026-11-24T14:3{i}:00Z") for i in range(4)]
    a = [Trade("AAPL", ts[0], 1, 10), Trade("AAPL", ts[2], 2, 10)]
    b = [Trade("MSFT", ts[1], 3, 10), Trade("MSFT", ts[3], 4, 10)]
    replay.write_csv(a, tmp_path / "a.csv")
    replay.write_csv(b, tmp_path / "b.csv")
    out = list(replay.replay(tmp_path / "a.csv", tmp_path / "b.csv"))
    assert [e.price for e in out] == [1, 3, 2, 4]


class _FakeWS:
    def __init__(self, frames):
        self.frames, self.sent = list(frames), []

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def recv(self): return self.frames.pop(0)
    async def send(self, m): self.sent.append(json.loads(m))
    def __aiter__(self): return self
    async def __anext__(self):
        if not self.frames:
            raise StopAsyncIteration
        return self.frames.pop(0)


def test_stream_auth_subscribe_and_events(monkeypatch):
    import asyncio
    import websockets
    frames = [
        json.dumps([{"T": "success", "msg": "connected"}]),
        json.dumps([{"T": "success", "msg": "authenticated"}]),
        json.dumps([{"T": "subscription", "trades": ["AAPL"]}]),
        json.dumps([{"T": "t", "S": "AAPL", "p": 230.0, "s": 5, "t": "2026-11-24T15:00:00Z"},
                    {"T": "t", "S": "AAPL", "p": 230.1, "s": 7, "t": "2026-11-24T15:00:01Z"}]),
    ]
    ws = _FakeWS(frames)
    monkeypatch.setattr(websockets, "connect", lambda url: ws)

    async def run():
        out = []
        async for ev in alpaca.stream(["AAPL"], key="k", secret="s", reconnect=False):
            out.append(ev)
        return out

    out = asyncio.run(run())
    assert [e.price for e in out] == [230.0, 230.1]
    assert ws.sent[0] == {"action": "auth", "key": "k", "secret": "s"}
    assert ws.sent[1] == {"action": "subscribe", "trades": ["AAPL"]}
