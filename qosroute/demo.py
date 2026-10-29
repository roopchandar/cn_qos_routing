"""Congestion-injection demo used by the dashboard and scripts/demo_congestion.py.

A "demo flow" (src -> dst) is tracked separately on top of optional background
traffic (the hotspot matrix). During [cong_start, cong_end) one link runs at a
reduced bandwidth, which makes it congested. We then compare, per algorithm,
which paths the demo flow's packets took before, during and after the event,
and what QoS they received.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from .experiment import make_router, run_episode
from .metrics import compute_metrics  # noqa: F401  (re-exported for convenience)
from .topology import build_topology
from .traffic import BASE_FLOWS, Flow, LinkEvent, Scenario


@dataclass
class DemoParams:
    src: int = 0
    dst: int = 9
    demo_rate_mbps: float = 4.0
    background: bool = True
    load: float = 0.6
    cong_link: tuple | None = (4, 7)
    cong_factor: float = 0.2
    cong_start: int = 700
    cong_end: int = 1400
    duration: int = 2000
    seed: int = 7
    window: int = 25

    def key(self):
        return tuple(sorted(asdict(self).items()))


def build_scenario(p: DemoParams) -> Scenario:
    flows = [Flow(f.src, f.dst, f.rate_mbps) for f in BASE_FLOWS] if p.background else []
    # The demo flow is always the LAST flow and is not scaled by the load slider.
    demo_rate = p.demo_rate_mbps / max(p.load, 1e-9)
    flows.append(Flow(p.src, p.dst, demo_rate))
    events = []
    if p.cong_link is not None and p.cong_factor < 1.0:
        u, v = p.cong_link
        events.append(LinkEvent(u, v, p.cong_start, p.cong_end, p.cong_factor))
    return Scenario("demo", flows, duration=p.duration, drain=300, load=p.load, events=events, seed=p.seed)


def _path_shares(log, flow, t0, t1):
    c = Counter(path for (tc, _, f, path, _) in log if f == flow and t0 <= tc < t1)
    total = sum(c.values())
    return [(list(path), n, n / total) for path, n in c.most_common()] if total else []


def _flow_stats(sim, flow, t0, t1):
    delays = [d for (tc, _, f, _, d) in sim.delivered_log if f == flow and t0 <= tc < t1]
    return {"delivered": len(delays),
            "avg_delay_ms": float(np.mean(delays)) if delays else float("nan")}


def run_demo(p: DemoParams, algorithm: str, graph=None, snapshot_every: int = 25) -> dict:
    graph = graph if graph is not None else build_topology()
    sc = build_scenario(p)
    sim, metrics = run_episode(sc, make_router(algorithm), graph, snapshot_every=snapshot_every)
    demo_flow = len(sc.flows) - 1

    has_event = bool(sc.events)
    periods = {"before": (0, p.cong_start), "during": (p.cong_start, p.cong_end),
               "after": (p.cong_end, p.duration)} if has_event else {"whole run": (0, p.duration)}

    w = p.window
    n = (sc.duration + sc.drain) // w
    delivered = sim.ts_delivered[: n * w].reshape(n, w).sum(1)
    delay_sum = sim.ts_delay_sum[: n * w].reshape(n, w).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_delay = np.where(delivered > 0, delay_sum / delivered, np.nan)
    demo_delay = np.full(n, np.nan)
    buckets: dict[int, list] = {}
    for (_, td, f, _, d) in sim.delivered_log:
        if f == demo_flow and td // w < n:
            buckets.setdefault(td // w, []).append(d)
    for b, ds in buckets.items():
        demo_delay[b] = np.mean(ds)

    demo_generated = sim.flow_generated[demo_flow]
    demo_delivered = sum(1 for e in sim.delivered_log if e[2] == demo_flow)
    return {
        "algorithm": algorithm,
        "metrics": metrics,
        "demo_flow": {"generated": demo_generated, "delivered": demo_delivered,
                      "pdr": demo_delivered / max(demo_generated, 1),
                      "avg_delay_ms": float(np.nanmean([e[4] for e in sim.delivered_log if e[2] == demo_flow]))
                      if demo_delivered else float("nan")},
        "periods": {name: {"window": rng, "paths": _path_shares(sim.delivered_log, demo_flow, *rng),
                           **_flow_stats(sim, demo_flow, *rng)} for name, rng in periods.items()},
        "timeseries": {"t": (np.arange(n) * w + w / 2).tolist(), "avg_delay": avg_delay.tolist(),
                       "demo_delay": demo_delay.tolist(),
                       "drops": sim.ts_dropped[: n * w].reshape(n, w).sum(1).tolist(),
                       "queued": sim.ts_queued[: n * w].reshape(n, w).mean(1).tolist()},
        "snapshots": [{"t": s.t, "occupancy": s.occupancy, "utilization": s.utilization}
                      for s in sim.snapshots],
        "events": [(e.u, e.v, e.start, e.end, e.factor) for e in sc.events],
    }


def describe(result: dict) -> str:
    """Plain-text summary (used by the CLI demo)."""
    lines = [f"=== {result['algorithm']} ==="]
    m = result["metrics"]
    lines.append(f"  network: delay {m['avg_delay_ms']:.1f} ms | PDR {m['pdr']:.3f} | "
                 f"loss {m['loss_ratio']:.3f} | throughput {m['throughput_mbps']:.2f} Mbps | "
                 f"jitter {m['jitter_ms']:.2f} ms")
    for name, per in result["periods"].items():
        lines.append(f"  demo flow {name:9s} (t={per['window'][0]}-{per['window'][1]} ms): "
                     f"avg delay {per['avg_delay_ms']:.1f} ms, delivered {per['delivered']}")
        for path, n, share in per["paths"][:3]:
            lines.append(f"      {share * 100:5.1f}%  " + " -> ".join(f"R{x}" for x in path))
    return "\n".join(lines)
