from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tickstream.broker import PAPER_URL, PaperBroker
from tickstream.events import Bar
from tickstream.pairs import backtest
from tickstream.paper import PairsTrader, Session, completed, orders_to_reach
from tickstream.synth import cointegrated_pair

T0 = datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc)   # 9:30 ET


def _bars(x, y, start=T0):
    out = []
    for i in range(len(x)):
        day, slot = divmod(i, 78)
        ts = start + timedelta(days=day, minutes=5 * slot)
        out += [Bar("Y", ts, 0, 0, 0, float(np.exp(y[i])), 1),
                Bar("X", ts, 0, 0, 0, float(np.exp(x[i])), 1)]
    return out


def test_live_rules_match_backtest_exactly():
    x, y = cointegrated_pair(n=3_000, seed=11)
    tr = PairsTrader("Y", "X", halflife=300)
    states, betas = [], []   # one bar at a time, as the live loop sees them
    for b in range(len(x)):
        tr.on_bars(_bars(x[b:b + 1], y[b:b + 1], T0 + timedelta(minutes=5 * b)))
        states.append(tr.s.state)
        betas.append(tr.s.held_beta)
    bt = backtest(x, y, halflife=300, warmup=0)
    assert states == bt.position.astype(int).tolist()
    expected, held = [], 0.0          # backtest freezes beta at each entry
    for t, p in enumerate(bt.position):
        if p != 0 and (t == 0 or bt.position[t - 1] != p):
            held = bt.beta[t]
        expected.append(held if p != 0 else 0.0)
    assert np.allclose(betas, expected)
    assert bt.trades > 3
    assert sum(1 for a, b in zip([0] + states, states) if a == 0 and b != 0) == bt.trades


def test_orders_cross_zero_in_two_steps():
    assert orders_to_reach({"A": -10, "B": 5}, {"A": 10}) == [("A", -10), ("A", -10), ("B", 5)]
    assert orders_to_reach({"A": 0, "B": 0}, {"A": 7, "B": -3, "Z": 1}) == [("A", -7), ("B", 3)]
    assert orders_to_reach({"A": 4}, {"A": 4}) == []


def test_target_shares_hedged_in_dollars():
    tr = PairsTrader("Y", "X", notional=10_000)
    tr.last_px = {"Y": 200.0, "X": 100.0}
    tr.s.state, tr.s.held_beta = 1, 0.9
    assert tr.target_shares() == {"Y": 50, "X": -90}     # long $10k Y, short $9k X
    tr.s.state = -1
    assert tr.target_shares() == {"Y": -50, "X": 90}


def test_completed_drops_unfinished_and_extended_hours():
    now = T0 + timedelta(minutes=12)
    bars = [Bar("A", T0 + timedelta(minutes=m), 0, 0, 0, 1, 1) for m in (0, 5, 10)]
    bars.append(Bar("A", T0 - timedelta(hours=2), 0, 0, 0, 1, 1))  # pre-market
    assert [b.ts for b in completed(bars, now, 5)] == [T0, T0 + timedelta(minutes=5)]


class FakeBroker:
    def __init__(self):
        self.pos, self.sent = {}, []

    def positions(self):
        return dict(self.pos)

    def market_order(self, sym, q):
        self.sent.append((sym, q))
        self.pos[sym] = self.pos.get(sym, 0) + q
        if self.pos[sym] == 0:
            del self.pos[sym]

    def equity(self):
        return 100_000.0


def test_session_trades_restarts_and_logs(tmp_path):
    x, y = cointegrated_pair(n=20 * 78, seed=3, reversion=0.03)
    allbars = _bars(x, y)
    fetch_calls = []

    def fetch(syms, start, upto):
        fetch_calls.append(start)
        return [b for b in allbars if start <= b.ts and b.ts + timedelta(minutes=5) <= upto]

    broker = FakeBroker()
    mid = allbars[len(allbars) // 2].ts
    mk = lambda: Session(PairsTrader("Y", "X", halflife=200), broker,
                         lambda s, st: fetch(s, st, clock["now"]),
                         tmp_path / "s.json", tmp_path / "log.csv", benchmark="Y")
    clock = {"now": mid}
    s = mk()
    s.warm_up(clock["now"], days=30)
    assert s.trader.s.state == 0 and s.trader.stats.count > 0
    # step forward bar by bar through the second half
    times = sorted({b.ts for b in allbars if b.ts > mid})
    for ts in times[:600]:
        clock["now"] = ts + timedelta(minutes=5, seconds=20)
        s.tick(clock["now"], market_open=True)
    assert broker.sent, "expected at least one trade on a cointegrated pair"
    tgt = s.trader.target_shares()
    assert {k: v for k, v in tgt.items() if v} == broker.positions()

    # restart: a fresh session resumes the saved position and timestamp
    saved = s.trader.s
    s2 = mk()
    s2.warm_up(clock["now"], days=30)
    assert s2.trader.s.state == saved.state
    assert s2.trader.s.last_ts == saved.last_ts
    assert s2.trader.beta == pytest.approx(s.trader.beta)
    assert (tmp_path / "log.csv").read_text().count("\n") > 10


def test_broker_only_talks_to_paper():
    calls = []

    def http(method, url, headers, body=None):
        calls.append((method, url, body))
        return {"id": "1"}

    b = PaperBroker(key="k", secret="s", http=http)
    b.market_order("GOOG", -7)
    method, url, body = calls[0]
    assert url == PAPER_URL + "/v2/orders" and "paper-api" in url
    assert body == {"symbol": "GOOG", "qty": "7", "side": "sell", "type": "market",
                    "time_in_force": "day"}
