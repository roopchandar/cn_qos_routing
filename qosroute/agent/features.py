"""State encoding and action masking for the DQN router.

Decision point: a packet sits at router u and must pick an outgoing link.
Action k = "send to the k-th neighbour of u" (neighbours sorted by id).
Routers have at most K = max_degree neighbours; unused slots are masked.

State vector (fixed length, all values roughly in [0, 1])
---------------------------------------------------------
Per action slot k (K slots x 11 features):
  0  slot_valid           1 if neighbour k exists
  1  queue_occupancy      packets in queue u->nb / queue capacity
  2  queue_delay          estimated wait (queue bytes / bandwidth), / 30 ms, clipped
  3  prop_delay           propagation delay / 5 ms
  4  bandwidth            current bandwidth / 10 Mbps
  5  hops_to_dst          min hop count nb->dst / 6
  6  prop_to_dst          min propagation delay nb->dst / 15 ms
  7  downstream_delay     at nb, best queue delay among links that move closer to
                          dst (/ 30 ms, clipped) - a one-step look-ahead an SDN
                          controller with a global view can provide
  8  is_destination       1 if nb == dst
  9  recent_drops         exponential moving average of drops on u->nb
  10 recent_utilisation   exponential moving average of utilisation of u->nb
Global features:
  one-hot(current router)  N values
  one-hot(destination)     N values
  remaining hop budget     (TTL - hops) / TTL

Action mask (True = allowed)
----------------------------
  neighbour exists AND neighbour not already visited by this packet (no loops)
  AND destination still reachable within the remaining TTL.
If that leaves nothing (dead end), revisits are allowed as a fallback.
"""
from __future__ import annotations

import numpy as np

N_SLOT_FEATURES = 11
QD_SCALE = 30.0


class StateEncoder:
    def __init__(self, sim):
        self.N = sim.n_nodes
        self.K = max(len(v) for v in sim.neighbors.values())
        self.ttl = sim.ttl
        self.link_keys = list(sim.links.keys())
        self.link_index = {k: i for i, k in enumerate(self.link_keys)}
        self.n_links = len(self.link_keys)
        pad = self.n_links  # index of an all-zero "padding" link

        self.NB = np.zeros((self.N, self.K), dtype=np.int64)
        self.L = np.full((self.N, self.K), pad, dtype=np.int64)
        self.VALID = np.zeros((self.N, self.K), dtype=bool)
        for u, nbs in sim.neighbors.items():
            for k, v in enumerate(nbs):
                self.NB[u, k] = v
                self.L[u, k] = self.link_index[(u, v)]
                self.VALID[u, k] = True

        self.HOP = np.array([[sim.hop_dist[a][b] for b in range(self.N)] for a in range(self.N)],
                            dtype=np.float32)
        self.PROPD = np.array([[sim.prop_dist[a][b] for b in range(self.N)] for a in range(self.N)],
                              dtype=np.float32)

        # For the downstream feature: links out of v that reduce hop distance to dst
        self.PROG = np.full((self.N, self.N, self.K), pad, dtype=np.int64)
        for v in range(self.N):
            for d in range(self.N):
                for k, w in enumerate(sim.neighbors[v]):
                    if self.HOP[w, d] < self.HOP[v, d]:
                        self.PROG[v, d, k] = self.link_index[(v, w)]

        self.state_dim = self.K * N_SLOT_FEATURES + 2 * self.N + 1
        self.eye = np.eye(self.N, dtype=np.float32)
        self.link_feat = None
        self.down = None

    def refresh(self, sim):
        """Read the current link state once per tick (vectorised afterwards)."""
        n = self.n_links
        lf = np.zeros((n + 1, 7), dtype=np.float32)
        for i, key in enumerate(self.link_keys):
            l = sim.links[key]
            lf[i, 0] = len(l.queue) / l.cap
            lf[i, 1] = min(l.queue_delay_ms / QD_SCALE, 1.0)
            lf[i, 2] = l.prop / 5.0
            lf[i, 3] = l.bw / 1250.0          # bytes/ms -> fraction of 10 Mbps
            lf[i, 4] = l.drop_ema
            lf[i, 5] = l.util_ema
        self.link_feat = lf
        qd = lf[:, 1].copy()
        qd[n] = np.inf                        # padding never wins the min
        down = qd[self.PROG].min(axis=2)      # (N, N)
        down[~np.isfinite(down)] = 1.0
        np.fill_diagonal(down, 0.0)
        self.down = down.astype(np.float32)

    def encode(self, packets):
        """Return (states [B, D], masks [B, K], neighbours [B, K], link ids [B, K])."""
        B = len(packets)
        U = np.fromiter((p.node for p in packets), dtype=np.int64, count=B)
        Dst = np.fromiter((p.dst for p in packets), dtype=np.int64, count=B)
        hops = np.fromiter((p.hops for p in packets), dtype=np.int64, count=B)
        visited = np.fromiter((p.visited for p in packets), dtype=np.int64, count=B)

        nb = self.NB[U]                       # (B, K)
        lk = self.L[U]
        valid = self.VALID[U]
        lf = self.link_feat[lk]               # (B, K, 7)
        d2 = Dst[:, None]

        slot = np.zeros((B, self.K, N_SLOT_FEATURES), dtype=np.float32)
        slot[..., 0] = valid
        slot[..., 1] = lf[..., 0]
        slot[..., 2] = lf[..., 1]
        slot[..., 3] = lf[..., 2]
        slot[..., 4] = lf[..., 3]
        slot[..., 5] = self.HOP[nb, d2] / 6.0
        slot[..., 6] = self.PROPD[nb, d2] / 15.0
        slot[..., 7] = self.down[nb, d2]
        slot[..., 8] = nb == d2
        slot[..., 9] = lf[..., 4]
        slot[..., 10] = lf[..., 5]
        slot *= valid[..., None]

        ttl_left = (self.ttl - hops).astype(np.float32)
        states = np.concatenate(
            [slot.reshape(B, -1), self.eye[U], self.eye[Dst], (ttl_left / self.ttl)[:, None]],
            axis=1,
        ).astype(np.float32)

        not_visited = ((visited[:, None] >> nb) & 1) == 0
        reachable = self.HOP[nb, d2] <= (ttl_left[:, None] - 1)
        mask = valid & not_visited & reachable
        empty = ~mask.any(axis=1)
        if empty.any():
            fb = valid & reachable
            mask[empty] = fb[empty]
            still = ~mask.any(axis=1)
            mask[still] = valid[still]
        return states, mask, nb, lk
