"""Baseline 2 - congestion-aware heuristic routing with dynamic link costs.

Every ``update_interval`` ms (think: a link-state update / SDN controller poll)
each directed link gets a cost

    cost(u->v) = propagation delay + estimated queueing delay + transmission time
               = prop_ms + queue_bytes / bandwidth + 1000 B / bandwidth
               (+ a penalty when the queue is nearly full)

and Dijkstra recomputes next-hop tables. Between updates the tables are
frozen, so the heuristic reacts to congestion with a delay (stale information),
which is realistic for link-state protocols.
"""
from __future__ import annotations

from .base import Router, next_hop_table


class CongestionAwareRouter(Router):
    name = "Congestion-Aware"

    def __init__(self, update_interval: int = 10, full_penalty: float = 20.0):
        self.update_interval = update_interval
        self.full_penalty = full_penalty
        self.nh = None

    def _recompute(self, sim):
        cost = {}
        for key, link in sim.links.items():
            bw = max(link.bw, 1e-6)
            c = link.prop + link.queue_bytes / bw + 1000.0 / bw
            if link.occupancy >= 0.9:
                c += self.full_penalty
            cost[key] = c
        self.nh = next_hop_table(cost, sim.nodes)

    def reset(self, sim):
        self._recompute(sim)

    def on_tick(self, sim):
        if sim.t % self.update_interval == 0:
            self._recompute(sim)

    def choose(self, sim, packets):
        nh = self.nh
        return [nh[p.node][p.dst] for p in packets]
