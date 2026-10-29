"""Baseline 1 - static shortest-path routing (OSPF-like with fixed link costs).

The cost of every link is its propagation delay, which never changes. Next-hop
tables are computed once with Dijkstra and every packet to a given destination
follows the same path regardless of congestion.
"""
from __future__ import annotations

from .base import Router, next_hop_table


class ShortestPathRouter(Router):
    name = "Shortest Path"

    def __init__(self, weight: str = "prop_ms"):
        self.weight = weight          # "prop_ms" (delay) or "hop" (hop count)
        self.nh = None

    def reset(self, sim):
        cost = {}
        for (u, v), link in sim.links.items():
            cost[(u, v)] = 1.0 if self.weight == "hop" else float(link.prop)
        self.nh = next_hop_table(cost, sim.nodes)

    def choose(self, sim, packets):
        nh = self.nh
        return [nh[p.node][p.dst] for p in packets]
