"""Common interface every routing algorithm implements.

The simulator calls, per tick:
    on_tick(sim)              - once, before decisions (e.g. refresh link costs)
    choose(sim, packets)      - batched: return one next-hop router per packet
and per packet:
    on_deliver(sim, pkt)      - packet reached its destination
    on_drop(sim, pkt, reason) - packet discarded ('overflow' or 'ttl')
and once at the end:
    on_episode_end(sim)
"""
from __future__ import annotations


class Router:
    name = "base"

    def reset(self, sim):
        pass

    def on_tick(self, sim):
        pass

    def choose(self, sim, packets):
        raise NotImplementedError

    def on_deliver(self, sim, pkt):
        pass

    def on_drop(self, sim, pkt, reason):
        pass

    def on_episode_end(self, sim):
        pass


def next_hop_table(digraph_cost, nodes):
    """Destination-rooted Dijkstra. ``digraph_cost[(u, v)]`` = cost of link u->v.

    Returns nh[u][dst] = neighbour of u on a minimum-cost path to dst.
    Ties are broken deterministically by the smaller neighbour id.
    """
    import heapq

    rev: dict[int, list[tuple[int, float]]] = {n: [] for n in nodes}
    for (u, v), c in digraph_cost.items():
        rev[v].append((u, c))
    nh = {u: {} for u in nodes}
    for dst in nodes:
        dist = {dst: 0.0}
        heap = [(0.0, dst)]
        done = set()
        while heap:
            d, v = heapq.heappop(heap)
            if v in done:
                continue
            done.add(v)
            for u, c in rev[v]:
                nd = d + c
                if nd < dist.get(u, float("inf")) - 1e-12 or (
                    abs(nd - dist.get(u, float("inf"))) <= 1e-12 and v < nh[u].get(dst, 1 << 30)
                ):
                    dist[u] = nd
                    nh[u][dst] = v
                    heapq.heappush(heap, (nd, u))
    return nh
