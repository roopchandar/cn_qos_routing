"""Experience replay buffer (fixed-size ring buffer of transitions).

A transition is (s, a, r, s', mask', done):
  s, a   - state and action at a router
  r      - reward observed for that hop
  s'     - state at the next router (ignored when done)
  mask'  - which actions are allowed at the next router (for max_a' Q(s', a'))
  done   - 1 if the packet was delivered or dropped
Sampling random mini-batches breaks the strong correlation between consecutive
routing decisions, which stabilises DQN training.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, n_actions: int, seed: int = 0):
        self.capacity = capacity
        self.s = np.zeros((capacity, state_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, state_dim), np.float32)
        self.m2 = np.zeros((capacity, n_actions), bool)
        self.done = np.zeros(capacity, np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, m2, done):
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        if s2 is not None:
            self.s2[i] = s2
            self.m2[i] = m2
        else:
            self.m2[i] = True
        self.done[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.size, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.m2[idx], self.done[idx]

    def __len__(self):
        return self.size
