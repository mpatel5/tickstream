"""Summarise a paper-trading log against SPY.

    python examples/paper_report.py                    # data/paper_GOOGL_GOOG.csv
    python examples/paper_report.py data/paper_XOM_CVX.csv
"""
import csv
import sys


def main(path="data/paper_GOOGL_GOOG.csv"):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["equity"]]
    if not rows:
        print("no rows logged yet")
        return
    first, last = rows[0], rows[-1]
    eq0, eq1 = float(first["equity"]), float(last["equity"])
    spy = [float(r["benchmark_px"]) for r in rows if r["benchmark_px"]]
    entries = sum(1 for a, b in zip(rows, rows[1:]) if a["state"] == "0" and b["state"] != "0")
    orders = sum(len(r["orders"].split()) for r in rows)
    print(f"{first['y']}/{first['x']}  {first['time'][:16]} -> {last['time'][:16]}  ({len(rows)} bars)")
    print(f"account equity  ${eq0:,.2f} -> ${eq1:,.2f}   {eq1 / eq0 - 1:+.3%}  (${eq1 - eq0:+,.2f})")
    if len(spy) >= 2:
        print(f"SPY             {spy[0]:.2f} -> {spy[-1]:.2f}   {spy[-1] / spy[0] - 1:+.3%}")
    print(f"trades opened   {entries}   orders sent {orders}   position now {int(last['state']):+d}"
          f"   z now {float(last['z']):+.2f}")
    print("\nNote: equity is the whole paper account, so it includes any manual trades.")


if __name__ == "__main__":
    main(*sys.argv[1:])
