"""Q-network implemented in pure NumPy (default backend).

Architecture: state -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(K)
Output k is Q(s, a=k): the expected discounted return of sending the packet to
neighbour k.

Training uses the Huber (smooth-L1) loss on Q(s, a) for the action actually
taken, back-propagation written by hand, gradient-norm clipping and Adam.
This file has no dependency on PyTorch so the project runs anywhere NumPy does.
"""
from __future__ import annotations

import numpy as np


class NumpyQNetwork:
    backend = "numpy"

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, lr: float = 1e-3,
                 seed: int = 0, grad_clip: float = 10.0):
        rng = np.random.default_rng(seed)
        self.in_dim, self.out_dim, self.hidden = in_dim, out_dim, hidden
        self.lr, self.grad_clip = lr, grad_clip

        def he(fan_in, fan_out):
            return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float32)

        self.params = [
            he(in_dim, hidden), np.zeros(hidden, np.float32),
            he(hidden, hidden), np.zeros(hidden, np.float32),
            (he(hidden, out_dim) * 0.1).astype(np.float32), np.zeros(out_dim, np.float32),
        ]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.step_count = 0

    # ----------------------------------------------------------------- forward
    def _forward(self, x):
        W1, b1, W2, b2, W3, b3 = self.params
        z1 = x @ W1 + b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ W2 + b2
        h2 = np.maximum(z2, 0.0)
        q = h2 @ W3 + b3
        return q, (x, z1, h1, z2, h2)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(x, dtype=np.float32))[0]

    # ----------------------------------------------------------------- training
    def gradients(self, x, actions, targets):
        """Huber loss on Q(s, a) and its gradient w.r.t. every parameter."""
        x = np.asarray(x, dtype=np.float32)
        q, (x, z1, h1, z2, h2) = self._forward(x)
        B = x.shape[0]
        idx = np.arange(B)
        diff = q[idx, actions] - targets
        absd = np.abs(diff)
        loss = float(np.where(absd < 1.0, 0.5 * diff ** 2, absd - 0.5).mean())

        dq = np.zeros_like(q)
        dq[idx, actions] = np.clip(diff, -1.0, 1.0) / B      # d(Huber)/dq
        W1, b1, W2, b2, W3, b3 = self.params
        gW3 = h2.T @ dq
        gb3 = dq.sum(0)
        dz2 = (dq @ W3.T) * (z2 > 0)
        gW2 = h1.T @ dz2
        gb2 = dz2.sum(0)
        dz1 = (dz2 @ W2.T) * (z1 > 0)
        gW1 = x.T @ dz1
        gb1 = dz1.sum(0)
        return loss, [gW1, gb1, gW2, gb2, gW3, gb3]

    def train_step(self, x, actions, targets) -> float:
        loss, grads = self.gradients(x, actions, targets)
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if norm > self.grad_clip:
            grads = [g * (self.grad_clip / norm) for g in grads]

        # Adam
        self.step_count += 1
        b1c, b2c = 0.9, 0.999
        t = self.step_count
        for i, g in enumerate(grads):
            self.m[i] = b1c * self.m[i] + (1 - b1c) * g
            self.v[i] = b2c * self.v[i] + (1 - b2c) * g * g
            mh = self.m[i] / (1 - b1c ** t)
            vh = self.v[i] / (1 - b2c ** t)
            self.params[i] -= (self.lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        return loss

    # ----------------------------------------------------------------- weights
    def get_weights(self) -> list[np.ndarray]:
        return [p.copy() for p in self.params]

    def set_weights(self, weights):
        self.params = [np.asarray(w, dtype=np.float32).copy() for w in weights]

    def soft_update(self, other, tau: float):
        """self <- tau * other + (1 - tau) * self  (used for the target network)."""
        for i, w in enumerate(other.get_weights()):
            self.params[i] = (tau * w + (1 - tau) * self.params[i]).astype(np.float32)
