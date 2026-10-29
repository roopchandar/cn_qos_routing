"""Terminal version of the congestion demo (no dashboard needed).

Usage (from the project root):
    python scripts/demo_congestion.py
    python scripts/demo_congestion.py --src 0 --dst 9 --link 4 7 --factor 0.2 --load 0.6

A demo flow runs from --src to --dst on top of background traffic. Between
--start and --end ms the link --link runs at --factor of its bandwidth. For
each algorithm the script prints which paths the demo flow used before,
during and after the congestion event, and the QoS it got.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qosroute.demo import DemoParams, describe, run_demo  # noqa: E402
from qosroute.experiment import ALGORITHMS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=int, default=0)
    ap.add_argument("--dst", type=int, default=9)
    ap.add_argument("--rate", type=float, default=4.0, help="demo flow rate (Mbps)")
    ap.add_argument("--load", type=float, default=0.6, help="background load multiplier")
    ap.add_argument("--no-background", action="store_true")
    ap.add_argument("--link", type=int, nargs=2, default=[4, 7])
    ap.add_argument("--factor", type=float, default=0.2, help="bandwidth fraction during event")
    ap.add_argument("--start", type=int, default=700)
    ap.add_argument("--end", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    p = DemoParams(src=a.src, dst=a.dst, demo_rate_mbps=a.rate, background=not a.no_background,
                   load=a.load, cong_link=tuple(a.link), cong_factor=a.factor,
                   cong_start=a.start, cong_end=a.end, seed=a.seed)
    print(f"Demo flow R{p.src} -> R{p.dst} at {p.demo_rate_mbps} Mbps; background load {p.load}; "
          f"link R{p.cong_link[0]}-R{p.cong_link[1]} at {p.cong_factor:.0%} bandwidth "
          f"during {p.cong_start}-{p.cong_end} ms\n")
    for algo in ALGORITHMS:
        print(describe(run_demo(p, algo)))
        print()


if __name__ == "__main__":
    main()
