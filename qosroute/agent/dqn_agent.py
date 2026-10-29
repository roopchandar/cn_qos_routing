"""Deep Q-Network agent: online net, target net, replay buffer, epsilon-greedy.

Learning rule (Double DQN target, masked):
    a*  = argmax_{a' allowed in s'} Q_online(s', a')
    y   = r + gamma * (1 - done) * Q_target(s', a*)
    loss = Huber(Q_online(s, a) - y)
The target network is a slowly-moving copy of the online network
(soft update: theta_target <- tau*theta + (1-tau)*theta_target), which keeps the
regression target from chasing itself.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .replay import ReplayBuffer


@dataclass
class DQNConfig:
    hidden: int = 128
    lr: float = 5e-4
    gamma: float = 0.98
    batch_size: int = 128
    buffer_size: int = 100_000
    learn_start: int = 2_000          # transitions collected before learning starts
    train_every: int = 1              # gradient steps happen every N ticks
    tau: float = 0.01                 # soft target-update rate
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 120_000    # ticks over which epsilon decays linearly
    seed: int = 0
    backend: str = "numpy"


def make_network(backend: str, in_dim: int, out_dim: int, hidden: int, lr: float, seed: int):
    if backend == "torch":
        from .qnet_torch import TorchQNetwork
        return TorchQNetwork(in_dim, out_dim, hidden=hidden, lr=lr, seed=seed)
    from .qnet_numpy import NumpyQNetwork
    return NumpyQNetwork(in_dim, out_dim, hidden=hidden, lr=lr, seed=seed)


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, cfg: DQNConfig | None = None):
        self.cfg = cfg or DQNConfig()
        c = self.cfg
        self.state_dim, self.n_actions = state_dim, n_actions
        self.online = make_network(c.backend, state_dim, n_actions, c.hidden, c.lr, c.seed)
        self.target = make_network(c.backend, state_dim, n_actions, c.hidden, c.lr, c.seed)
        self.target.set_weights(self.online.get_weights())
        self.buffer = ReplayBuffer(c.buffer_size, state_dim, n_actions, seed=c.seed)
        self.rng = np.random.default_rng(c.seed + 1)
        self.ticks = 0
        self.losses: list[float] = []

    # ------------------------------------------------------------------ acting
    @property
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(self.ticks / max(c.eps_decay_steps, 1), 1.0)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, states, masks, explore: bool):
        q = self.online.predict(states)
        q = np.where(masks, q, -np.inf)
        actions = q.argmax(axis=1)
        if explore:
            eps = self.epsilon
            rand = self.rng.random(len(actions)) < eps
            for i in np.nonzero(rand)[0]:
                allowed = np.nonzero(masks[i])[0]
                actions[i] = self.rng.choice(allowed)
        return actions, q

    # ------------------------------------------------------------------ learning
    def tick(self):
        """Called once per simulated ms while training."""
        self.ticks += 1
        c = self.cfg
        if len(self.buffer) < c.learn_start or self.ticks % c.train_every:
            return None
        s, a, r, s2, m2, done = self.buffer.sample(c.batch_size)
        q_next_online = np.where(m2, self.online.predict(s2), -np.inf)
        a_star = q_next_online.argmax(axis=1)
        q_next_target = self.target.predict(s2)[np.arange(len(a_star)), a_star]
        y = r + c.gamma * (1.0 - done) * q_next_target
        loss = self.online.train_step(s, a, y.astype(np.float32))
        self.target.soft_update(self.online, c.tau)
        self.losses.append(loss)
        return loss

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path, meta: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        w = self.online.get_weights()
        info = {"state_dim": self.state_dim, "n_actions": self.n_actions,
                "config": asdict(self.cfg), **(meta or {})}
        np.savez(path, *w, meta=json.dumps(info))

    @classmethod
    def load(cls, path: str | Path, backend: str | None = None) -> "DQNAgent":
        data = np.load(path, allow_pickle=False)
        info = json.loads(str(data["meta"]))
        cfg = DQNConfig(**info["config"])
        if backend:
            cfg.backend = backend
        agent = cls(info["state_dim"], info["n_actions"], cfg)
        weights = [data[f"arr_{i}"] for i in range(6)]
        agent.online.set_weights(weights)
        agent.target.set_weights(weights)
        agent.meta = info
        return agent
