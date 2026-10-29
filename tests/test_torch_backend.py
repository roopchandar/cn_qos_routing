"""Checks that the optional PyTorch backend matches the NumPy backend.
Skipped automatically if PyTorch is not installed.
    python -m pytest tests/test_torch_backend.py -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from qosroute.agent.qnet_numpy import NumpyQNetwork  # noqa: E402


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchBackend(unittest.TestCase):
    def setUp(self):
        from qosroute.agent.qnet_torch import TorchQNetwork
        self.np_net = NumpyQNetwork(10, 4, hidden=16, lr=1e-3, seed=0)
        self.t_net = TorchQNetwork(10, 4, hidden=16, lr=1e-3, seed=0)
        self.t_net.set_weights(self.np_net.get_weights())
        rng = np.random.default_rng(0)
        self.x = rng.standard_normal((32, 10)).astype(np.float32)
        self.a = rng.integers(0, 4, 32)
        self.y = rng.standard_normal(32).astype(np.float32)

    def test_forward_matches(self):
        np.testing.assert_allclose(self.np_net.predict(self.x), self.t_net.predict(self.x),
                                   rtol=1e-4, atol=1e-5)

    def test_one_training_step_matches(self):
        l1 = self.np_net.train_step(self.x, self.a, self.y)
        l2 = self.t_net.train_step(self.x, self.a, self.y)
        self.assertAlmostEqual(l1, l2, places=4)
        for w1, w2 in zip(self.np_net.get_weights(), self.t_net.get_weights()):
            np.testing.assert_allclose(w1, w2, rtol=1e-3, atol=1e-5)

    def test_saved_model_loads_in_torch(self):
        from qosroute.agent.dqn_agent import DQNAgent
        from qosroute.experiment import DEFAULT_MODEL
        if not DEFAULT_MODEL.exists():
            self.skipTest("no trained model")
        a_np = DQNAgent.load(DEFAULT_MODEL, backend="numpy")
        a_t = DQNAgent.load(DEFAULT_MODEL, backend="torch")
        x = np.random.default_rng(1).random((8, a_np.state_dim)).astype(np.float32)
        np.testing.assert_allclose(a_np.online.predict(x), a_t.online.predict(x), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
