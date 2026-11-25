"""Paper-trade a pair on your Alpaca paper account, every 5 minutes.

    caffeinate -i python examples/paper_trader.py              # GOOGL / GOOG
    caffeinate -i python examples/paper_trader.py --y XOM --x CVX --notional 5000

`caffeinate -i` keeps the Mac awake. Leave it running; it sleeps while the
market is closed and wakes at the open. Ctrl+C stops it, and restarting picks
up where it left off (state in data/paper_<Y>_<X>.json). Every bar is logged
to data/paper_<Y>_<X>.csv; `python examples/paper_report.py` summarises it.

Paper money only: tickstream.broker can't reach a live account.
"""
import argparse
import time
from datetime import timedelta
from pathlib import Path

from tickstream.broker import PaperBroker
from tickstream.feeds import alpaca
from tickstream.paper import PairsTrader, Session, utcnow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", default="GOOGL")
    ap.add_argument("--x", default="GOOG")
    ap.add_argument("--notional", type=float, default=10_000, help="dollars in the y leg")
    ap.add_argument("--halflife", type=float, default=390, help="bars (390 x 5 min = 5 days)")
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--exit", type=float, default=0.5)
    ap.add_argument("--stop", type=float, default=4.0)
    a = ap.parse_args()

    tag = f"{a.y}_{a.x}"
    broker = PaperBroker()
    fetch = lambda syms, start: list(alpaca.historical_bars(syms, start=start, timeframe="5Min"))
    trader = PairsTrader(a.y, a.x, halflife=a.halflife, entry_z=a.entry, exit_z=a.exit,
                         stop_z=a.stop, notional=a.notional)
    s = Session(trader, broker, fetch, Path(f"data/paper_{tag}.json"),
                Path(f"data/paper_{tag}.csv"), max_leg_notional=2.5 * a.notional)

    n = s.warm_up(utcnow())
    print(f"{a.y}/{a.x}: warmed up on {n:,} bars | beta {trader.beta:.3f} | "
          f"z {trader.z:+.2f} | position {trader.s.state:+d} | "
          f"account equity ${broker.equity():,.2f}")

    while True:
        try:
            clock = broker.clock()
            now = clock.now
            if not clock.is_open:
                wait = (clock.next_open - now).total_seconds()
                print(f"{now:%a %H:%M} UTC market closed; next open "
                      f"{clock.next_open:%a %Y-%m-%d %H:%M} UTC ({wait / 3600:.1f}h)")
                s.tick(now, market_open=False)      # keep estimators current
                time.sleep(max(60, min(wait + 60, 3600)))
                continue
            orders = s.tick(now, market_open=True)
            t = trader
            print(f"{now:%H:%M} z={t.z:+.2f} beta={t.beta:.3f} pos={t.s.state:+d} "
                  f"{t.y}={t.last_px.get(t.y, 0):.2f} {t.x}={t.last_px.get(t.x, 0):.2f}"
                  + (f"  ORDERS {orders}" if orders else ""))
        except KeyboardInterrupt:
            raise
        except Exception as e:                      # network blip, rejected order, ...
            print(f"error: {e} (retrying next bar)")
        # wake 20 s after the next 5-minute boundary, when that bar is complete
        now = utcnow()
        nxt = now.replace(second=20, microsecond=0) + timedelta(minutes=5 - now.minute % 5)
        time.sleep(max(5, (nxt - now).total_seconds()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped; state saved")
