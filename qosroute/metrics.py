"""QoS metric definitions (all computed the same way for every algorithm).

Accounting
----------
Only packets *generated* during the traffic window [0, duration) are counted.
After the window the simulator keeps running for ``drain`` ms with no new
traffic so queued packets can finish. Every generated packet ends in exactly one
of three states:

    delivered + dropped (queue overflow or TTL) + still in flight = generated

Definitions
-----------
avg_delay_ms      mean end-to-end delay of DELIVERED packets only (ms).
                  Dropped packets have no delay; they are reflected in loss.
p95_delay_ms      95th percentile of delivered-packet delay.
jitter_ms         per flow, mean |d_i - d_(i-1)| over consecutive delivered
                  packets (delivery order), averaged over flows weighted by
                  packet count (RFC 3550 idea, without smoothing).
throughput_mbps   delivered bytes x 8 / traffic window (goodput, Mbps).
pdr               delivered / generated.
loss_ratio        (overflow drops + TTL drops) / generated.
undelivered_ratio still-in-flight-at-end / generated.
avg_link_util     mean over directed links of transmitted / available bytes.
max_link_util     the busiest directed link's utilisation.
congestion_freq   fraction of (link, ms) samples with queue >= 80 % full.
avg_hops          mean hop count of delivered packets.
"""
from __future__ import annotations

import numpy as np

METRIC_LABELS = {
    "avg_delay_ms": "Avg delay (ms)",
    "p95_delay_ms": "95th pct delay (ms)",
    "jitter_ms": "Jitter (ms)",
    "throughput_mbps": "Throughput (Mbps)",
    "pdr": "Packet delivery ratio",
    "loss_ratio": "Packet loss ratio",
    "undelivered_ratio": "Undelivered ratio",
    "avg_link_util": "Avg link utilisation",
    "max_link_util": "Max link utilisation",
    "congestion_freq": "Congestion frequency",
    "avg_hops": "Avg hops",
}


def jitter(flow_delays: dict[int, list[int]]) -> float:
    total, weight = 0.0, 0
    for delays in flow_delays.values():
        if len(delays) < 2:
            continue
        d = np.asarray(delays, dtype=float)
        total += np.abs(np.diff(d)).mean() * (len(d) - 1)
        weight += len(d) - 1
    return total / weight if weight else 0.0


def compute_metrics(sim) -> dict:
    gen = max(sim.generated, 1)
    delays = np.asarray(sim.delays, dtype=float)
    dur_ms = sim.scenario.duration
    utils = [l.tx_bytes / l.bw_sum for l in sim.link_list if l.bw_sum > 0]
    return {
        "generated": sim.generated,
        "delivered": sim.delivered,
        "dropped_overflow": sim.dropped_overflow,
        "dropped_ttl": sim.dropped_ttl,
        "in_flight": sim.in_flight,
        "offered_mbps": sim.generated_bytes * 8 / (dur_ms * 1000),
        "avg_delay_ms": float(delays.mean()) if delays.size else float("nan"),
        "p95_delay_ms": float(np.percentile(delays, 95)) if delays.size else float("nan"),
        "jitter_ms": jitter(sim.flow_delays),
        "throughput_mbps": sim.delivered_bytes * 8 / (dur_ms * 1000),
        "pdr": sim.delivered / gen,
        "loss_ratio": (sim.dropped_overflow + sim.dropped_ttl) / gen,
        "undelivered_ratio": sim.in_flight / gen,
        "avg_link_util": float(np.mean(utils)) if utils else 0.0,
        "max_link_util": float(np.max(utils)) if utils else 0.0,
        "congestion_freq": sim.congested_samples / max(sim.congestion_samples, 1),
        "avg_hops": sim.total_hops / max(sim.delivered, 1),
    }
