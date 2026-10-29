"""DQN-based adaptive routing: one shared Q-network decides every hop.

Each router asks the same trained network "which neighbour should this packet
go to next?" using the local link state plus a one-step downstream look-ahead
(see agent/features.py). In training mode the router also turns every hop into
a reinforcement-learning transition.

Reward (per hop, assigned when the outcome of the hop is known)
---------------------------------------------------------------
    r = 10 * delivered - 0.1 * hop_delay_ms - 5 * dropped - 3 * congestion_penalty

    hop_delay_ms        time from the decision at u until arrival at the next
                        router (queueing + transmission + propagation)
    delivered           1 if the next router is the destination
    dropped             1 if the packet was dropped (queue overflow or TTL)
    congestion_penalty  max(0, occupancy - 0.7) / 0.3 of the chosen queue at
                        decision time, in [0, 1]

Summed over a packet's path the delay terms give -0.1 x end-to-end delay, so
maximising return = deliver the packet, quickly, without filling queues.
"""
from __future__ import annotations

import numpy as np

from ..agent.features import StateEncoder
from .base import Router

R_DELIVER = 10.0
R_DELAY = 0.1
R_DROP = 5.0
R_CONGESTION = 3.0


def congestion_penalty(occupancy: np.ndarray) -> np.ndarray:
    return np.clip((occupancy - 0.7) / 0.3, 0.0, 1.0)


class DQNRouter(Router):
    name = "DQN"

    def __init__(self, agent, train: bool = False):
        self.agent = agent
        self.train = train
        self.encoder = None
        self.episode_reward = 0.0
        self.decisions = 0
        self.last_q = None

    def reset(self, sim):
        self.encoder = StateEncoder(sim)
        if self.encoder.state_dim != self.agent.state_dim or self.encoder.K != self.agent.n_actions:
            raise ValueError("Model was trained for a different topology/state size "
                             f"(model {self.agent.state_dim}/{self.agent.n_actions}, "
                             f"sim {self.encoder.state_dim}/{self.encoder.K}). Retrain the model.")
        self.episode_reward = 0.0
        self.decisions = 0

    def on_tick(self, sim):
        self.encoder.refresh(sim)
        if self.train:
            self.agent.tick()

    def _store(self, pkt, reward, s2=None, m2=None, done=False):
        self.agent.buffer.add(pkt.rl_state, pkt.rl_action, reward, s2, m2, done)
        self.episode_reward += reward

    def choose(self, sim, packets):
        states, masks, nb, lk = self.encoder.encode(packets)
        actions, q = self.agent.act(states, masks, explore=self.train)
        occ = self.encoder.link_feat[lk[np.arange(len(packets)), actions], 0]
        pens = congestion_penalty(occ)
        rows = np.arange(len(packets))
        next_hops = nb[rows, actions]
        t = sim.t
        for i, pkt in enumerate(packets):
            if self.train and pkt.rl_state is not None:
                # finish the previous hop's transition: we now know s'
                r = -R_DELAY * (t - pkt.t_decision) - R_CONGESTION * pkt.rl_penalty
                self._store(pkt, r, states[i], masks[i], done=False)
            pkt.rl_state = states[i] if self.train else None
            pkt.rl_action = int(actions[i])
            pkt.rl_penalty = float(pens[i])
        self.decisions += len(packets)
        self.last_q = q
        return next_hops.tolist()

    def on_deliver(self, sim, pkt):
        if self.train and pkt.rl_state is not None:
            r = R_DELIVER - R_DELAY * (sim.t - pkt.t_decision) - R_CONGESTION * pkt.rl_penalty
            self._store(pkt, r, done=True)

    def on_drop(self, sim, pkt, reason):
        if self.train and pkt.rl_state is not None:
            r = -R_DROP - R_CONGESTION * pkt.rl_penalty
            self._store(pkt, r, done=True)
