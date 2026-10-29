"""Optional PyTorch backend with the same interface as NumpyQNetwork.

Select it with ``--backend torch`` in scripts/train.py / scripts/evaluate.py.
Weights are stored in the same layout ([W1, b1, W2, b2, W3, b3] with W of shape
(in, out)), so a model trained with either backend can be loaded by the other.
Run ``pytest tests/test_torch_backend.py`` to check both backends agree.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - only hit when torch is missing
    raise ImportError("PyTorch backend requested but torch is not installed. "
                      "Run: pip install torch") from exc


def pick_device(prefer: str = "cpu") -> "torch.device":
    # For a network this small, CPU is usually faster than Apple's MPS GPU
    # (kernel-launch overhead dominates). Pass prefer="mps" to try the GPU.
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))


class TorchQNetwork:
    backend = "torch"

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, lr: float = 1e-3,
                 seed: int = 0, grad_clip: float = 10.0, device: str = "cpu"):
        torch.manual_seed(seed)
        self.in_dim, self.out_dim, self.hidden = in_dim, out_dim, hidden
        self.grad_clip = grad_clip
        self.device = pick_device(device)
        self.model = _MLP(in_dim, out_dim, hidden).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
            return self.model(t).cpu().numpy()

    def train_step(self, x, actions, targets) -> float:
        xt = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        at = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=self.device)
        yt = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=self.device)
        q_sa = self.model(xt).gather(1, at[:, None]).squeeze(1)
        loss = self.loss_fn(q_sa, yt)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        return float(loss.item())

    def get_weights(self) -> list[np.ndarray]:
        out = []
        for layer in (self.model.l1, self.model.l2, self.model.l3):
            out.append(layer.weight.detach().cpu().numpy().T.copy())   # (in, out)
            out.append(layer.bias.detach().cpu().numpy().copy())
        return out

    def set_weights(self, weights):
        with torch.no_grad():
            for i, layer in enumerate((self.model.l1, self.model.l2, self.model.l3)):
                layer.weight.copy_(torch.as_tensor(np.asarray(weights[2 * i]).T.copy()))
                layer.bias.copy_(torch.as_tensor(np.asarray(weights[2 * i + 1])))

    def soft_update(self, other, tau: float):
        mine = self.get_weights()
        theirs = other.get_weights()
        self.set_weights([tau * b + (1 - tau) * a for a, b in zip(mine, theirs)])
