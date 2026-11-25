"""Live pairs trading on an Alpaca paper account.

Runs the same rules as tickstream.pairs.backtest, one 5-minute bar at a time:
  - PairStats (C++) scores each bar against earlier bars only
  - short the spread when z > entry_z, buy it when z < -entry_z,
    exit inside +/- exit_z, stop out beyond +/- stop_z
  - hedge ratio frozen at entry; x leg sized at beta * the y leg's dollars

State (position side, frozen beta, last bar processed) is saved to JSON after
every bar, so the trader can be stopped and restarted.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ._core import PairStats
from .bars import regular_hours
from .events import Bar


@dataclass
class TraderState:
    state: int = 0                 # +1 long spread (long y), -1 short spread, 0 flat
    held_beta: float = 0.0
    last_ts: str | None = None     # ISO timestamp of the last bar processed
    trades: int = 0


class PairsTrader:
    def __init__(self, y: str, x: str, halflife: float = 390, entry_z: float = 2.0,
                 exit_z: float = 0.5, stop_z: float = 4.0, notional: float = 10_000):
        self.y, self.x = y, x
        self.entry_z, self.exit_z, self.stop_z = entry_z, exit_z, stop_z
        self.notional = notional
        self.stats = PairStats(halflife)
        self.s = TraderState()
        self.last_px: dict[str, float] = {}
        self.z = 0.0
        self.beta = 0.0

    # -- signal --------------------------------------------------------------
    def _rules(self) -> None:
        z, st = self.z, self.s
        if st.state == 0:
            if self.entry_z < z < self.stop_z:
                st.state, st.held_beta, st.trades = -1, self.beta, st.trades + 1
            elif -self.stop_z < z < -self.entry_z:
                st.state, st.held_beta, st.trades = 1, self.beta, st.trades + 1
        elif abs(z) < self.exit_z or abs(z) > self.stop_z:
            st.state, st.held_beta = 0, 0.0

    def on_bars(self, bars: Iterable[Bar], trade: bool = True) -> int:
        """Feed bars (any order, both symbols); process each new 5-min timestamp.

        With trade=False the estimators warm up but the position rules don't run.
        Returns the number of timestamps processed.
        """
        by_ts: dict[datetime, dict[str, float]] = {}
        for b in bars:
            if b.symbol in (self.y, self.x):
                by_ts.setdefault(b.ts, {})[b.symbol] = b.close
        last = datetime.fromisoformat(self.s.last_ts) if self.s.last_ts else None
        n = 0
        for ts in sorted(by_ts):
            if last is not None and ts <= last:
                continue
            self.last_px.update(by_ts[ts])
            if self.y not in self.last_px or self.x not in self.last_px:
                continue
            self.beta, _, self.z = self.stats.update(math.log(self.last_px[self.x]),
                                                     math.log(self.last_px[self.y]))
            if trade:
                self._rules()
            self.s.last_ts = ts.isoformat()
            n += 1
        return n

    # -- sizing ----------------------------------------------------------------
    def target_shares(self) -> dict[str, int]:
        st = self.s.state
        if st == 0 or not self.last_px:
            return {self.y: 0, self.x: 0}
        qy = st * round(self.notional / self.last_px[self.y])
        qx = -st * round(self.s.held_beta * self.notional / self.last_px[self.x])
        return {self.y: int(qy), self.x: int(qx)}

    # -- persistence -----------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self.s), indent=2))

    def load(self, path: Path) -> None:
        if path.exists():
            self.s = TraderState(**json.loads(path.read_text()))


def orders_to_reach(target: dict[str, int], current: dict[str, int]) -> list[tuple[str, int]]:
    """Signed share orders that move `current` to `target`.

    A position that changes side (long -> short) is closed first and reopened
    in a second order, because a single order can't cross zero.
    """
    out = []
    for sym, want in target.items():
        have = current.get(sym, 0)
        if want == have:
            continue
        if have and want and (have > 0) != (want > 0):
            out += [(sym, -have), (sym, want)]
        else:
            out.append((sym, want - have))
    return out


def completed(bars: Iterable[Bar], now: datetime, minutes: int) -> list[Bar]:
    """Regular-hours bars whose interval has fully closed by `now`."""
    width = timedelta(minutes=minutes)
    return [b for b in regular_hours(bars) if b.ts + width <= now]


@dataclass
class Session:
    """One trader + broker + data source; `tick()` is called every few minutes."""
    trader: PairsTrader
    broker: object                                   # PaperBroker or a test double
    fetch: Callable[[Sequence[str], datetime], list[Bar]]  # (symbols, start) -> 5-min bars
    state_path: Path
    log_path: Path
    minutes: int = 5
    max_leg_notional: float = 25_000
    benchmark: str = "SPY"
    spy_px: float | None = field(default=None, init=False)

    def warm_up(self, now: datetime, days: int = 21) -> int:
        """Load saved state, then run the estimators over recent history."""
        self.trader.load(self.state_path)
        saved = self.trader.s
        self.trader.s = type(saved)()                # replay history without trading
        bars = completed(self.fetch([self.trader.y, self.trader.x], now - timedelta(days=days)),
                         now, self.minutes)
        cutoff = datetime.fromisoformat(saved.last_ts) if saved.last_ts else now
        n = self.trader.on_bars([b for b in bars if b.ts <= cutoff], trade=False)
        last_ts = self.trader.s.last_ts
        self.trader.s = saved
        self.trader.s.last_ts = last_ts
        # bars after the saved timestamp (missed while stopped) run through the rules
        return n + self.trader.on_bars([b for b in bars if b.ts > cutoff])

    def tick(self, now: datetime, market_open: bool) -> list[tuple[str, int]]:
        t = self.trader
        since = datetime.fromisoformat(t.s.last_ts) if t.s.last_ts else now - timedelta(days=1)
        bars = completed(self.fetch([t.y, t.x, self.benchmark], since - timedelta(minutes=30)),
                         now, self.minutes)
        spy = [b for b in bars if b.symbol == self.benchmark]
        if spy:
            self.spy_px = spy[-1].close
        new = t.on_bars(bars)
        orders: list[tuple[str, int]] = []
        if market_open:
            target = t.target_shares()
            for sym, q in target.items():
                if abs(q) * t.last_px.get(sym, 0) > self.max_leg_notional:
                    raise RuntimeError(f"{sym} target {q} shares exceeds the per-leg cap")
            orders = orders_to_reach(target, self.broker.positions())
            for sym, q in orders:
                self.broker.market_order(sym, q)
        t.save(self.state_path)
        if new or orders:
            self._log(now, orders)
        return orders

    def _log(self, now: datetime, orders) -> None:
        t = self.trader
        new_file = not self.log_path.exists()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["time", "bar", "y", "x", "py", "px", "z", "beta", "state",
                            "orders", "equity", "benchmark_px"])
            w.writerow([now.isoformat(), t.s.last_ts, t.y, t.x, t.last_px.get(t.y),
                        t.last_px.get(t.x), round(t.z, 4), round(t.beta, 4), t.s.state,
                        " ".join(f"{s}:{q:+d}" for s, q in orders), self.broker.equity(),
                        self.spy_px])


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
