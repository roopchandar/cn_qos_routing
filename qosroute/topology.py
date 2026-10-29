"""Network topology: routers are graph nodes, links are edges.

Each physical link is full-duplex, so the simulator turns every undirected edge
into two *directed* links (u->v and v->u), each with its own output queue.

Link attributes
---------------
bw_mbps   : bandwidth in megabits per second
prop_ms   : propagation delay in milliseconds (= simulator ticks)
queue_cap : maximum number of packets the output queue can hold
"""
from __future__ import annotations

import networkx as nx

# (u, v, bandwidth Mbps, propagation delay ms)
# 10 routers, 15 links. Designed so that several distinct paths exist between
# most source/destination pairs and so that the lowest-delay path between the
# busiest pairs is NOT the only (or the highest-capacity) option.
DEFAULT_LINKS = [
    (0, 1, 10, 2),
    (0, 2, 10, 2),
    (0, 3, 8, 3),
    (1, 2, 5, 1),
    (1, 4, 10, 2),
    (2, 3, 5, 2),
    (2, 5, 10, 3),
    (3, 6, 8, 3),
    (4, 5, 5, 2),
    (4, 7, 10, 2),
    (5, 7, 8, 2),
    (5, 8, 10, 3),
    (6, 8, 8, 3),
    (7, 9, 10, 2),
    (8, 9, 10, 3),
]

# Fixed 2-D positions for drawing (left = sources side, right = sinks side)
DEFAULT_POS = {
    0: (0.0, 0.5),
    1: (1.0, 1.0),
    2: (1.0, 0.5),
    3: (1.0, 0.0),
    4: (2.0, 1.0),
    5: (2.0, 0.5),
    6: (2.0, 0.0),
    7: (3.0, 0.85),
    8: (3.0, 0.15),
    9: (4.0, 0.5),
}

DEFAULT_QUEUE_CAP = 30  # packets per directed output queue


def build_topology(links=None, queue_cap: int = DEFAULT_QUEUE_CAP) -> nx.Graph:
    """Return an undirected NetworkX graph with link attributes."""
    links = DEFAULT_LINKS if links is None else links
    g = nx.Graph()
    for u, v, bw, prop in links:
        if prop < 1:
            raise ValueError("propagation delay must be >= 1 ms (one tick)")
        g.add_edge(u, v, bw_mbps=float(bw), prop_ms=int(prop), queue_cap=int(queue_cap))
    if not nx.is_connected(g):
        raise ValueError("topology must be connected")
    return g


def neighbor_table(g: nx.Graph) -> dict[int, list[int]]:
    """Sorted neighbour list per router. Index k in this list = DQN action k."""
    return {n: sorted(g.neighbors(n)) for n in sorted(g.nodes)}


def max_degree(g: nx.Graph) -> int:
    return max(dict(g.degree()).values())


def hop_distances(g: nx.Graph) -> dict[int, dict[int, int]]:
    """All-pairs minimum hop count."""
    return {s: dict(d) for s, d in nx.all_pairs_shortest_path_length(g)}


def prop_distances(g: nx.Graph) -> dict[int, dict[int, float]]:
    """All-pairs minimum propagation delay (fixed link cost)."""
    return {s: dict(d) for s, d in nx.all_pairs_dijkstra_path_length(g, weight="prop_ms")}


def mbps_to_bytes_per_ms(mbps: float) -> float:
    # 1 Mbps = 1e6 bit/s = 125,000 byte/s = 125 byte/ms
    return mbps * 125.0
