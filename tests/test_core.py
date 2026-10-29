"""Core correctness tests. Run from the project root with either:
    python -m pytest -q
    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qosroute.agent.features import StateEncoder  # noqa: E402
from qosroute.agent.qnet_numpy import NumpyQNetwork  # noqa: E402
from qosroute.agent.replay import ReplayBuffer  # noqa: E402
from qosroute.experiment import DEFAULT_MODEL, make_router, run_episode  # noqa: E402
from qosroute.metrics import jitter  # noqa: E402
from qosroute.routing import CongestionAwareRouter, ShortestPathRouter  # noqa: E402
from qosroute.simulator import Simulator  # noqa: E402
from qosroute.topology import build_topology, max_degree  # noqa: E402
from qosroute.traffic import BASE_FLOWS, Flow, LinkEvent, Scenario, named_scenario  # noqa: E402

G = build_topology()
HAS_MODEL = DEFAULT_MODEL.exists()
ALGOS = ["Shortest Path", "Congestion-Aware"] + (["DQN"] if HAS_MODEL else [])


class TestTopology(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(G.number_of_nodes(), 10)
        self.assertEqual(G.number_of_edges(), 15)
        self.assertEqual(max_degree(G), 4)

    def test_link_attributes(self):
        for _, _, d in G.edges(data=True):
            self.assertGreater(d["bw_mbps"], 0)
            self.assertGreaterEqual(d["prop_ms"], 1)


class TestSimulator(unittest.TestCase):
    def test_packet_conservation(self):
        """generated = delivered + dropped + still in flight, for every algorithm."""
        for algo in ALGOS:
            sim, m = run_episode(named_scenario("heavy", seed=3, duration=800), make_router(algo), G)
            total = m["delivered"] + m["dropped_overflow"] + m["dropped_ttl"] + m["in_flight"]
            self.assertEqual(total, m["generated"], algo)
            self.assertAlmostEqual(m["pdr"] + m["loss_ratio"] + m["undelivered_ratio"], 1.0, places=9)

    def test_same_traffic_for_all_algorithms(self):
        """Common random numbers: identical arrivals whatever the routing."""
        gens = {algo: run_episode(named_scenario("moderate", seed=11, duration=500),
                                  make_router(algo), G)[1]["generated"] for algo in ALGOS}
        self.assertEqual(len(set(gens.values())), 1, gens)

    def test_delay_lower_bound(self):
        """No packet can arrive faster than the minimum propagation delay."""
        for algo in ALGOS:
            sim, _ = run_episode(named_scenario("moderate", seed=5, duration=500), make_router(algo), G)
            for (_, _, flow, path, delay) in sim.delivered_log:
                f = sim.scenario.flows[flow]
                self.assertGreaterEqual(delay, sim.prop_dist[f.src][f.dst])
                self.assertEqual(path[0], f.src)
                self.assertEqual(path[-1], f.dst)

    def test_single_packet_delay_exact(self):
        """One packet on an idle network: delay = sum of propagation delays."""
        sc = Scenario("one", [Flow(0, 9, 0.0)], duration=5, drain=50)
        sim = Simulator(G, sc, ShortestPathRouter())
        from qosroute.simulator import Packet
        pkt = Packet(0, 0, 0, 9, 1000, 0)
        sim.in_network = 1
        sim.generated = 1
        sim._generate = lambda: [pkt] if sim.t == 0 else []
        sim.run()
        self.assertEqual(sim.delivered, 1)
        self.assertEqual(sim.delays[0], sim.prop_dist[0][9])

    def test_queue_overflow_drops(self):
        """A tiny-bandwidth bottleneck must overflow and drop packets."""
        sc = Scenario("ovf", [Flow(0, 1, 20.0)], duration=400, drain=50,
                      events=[LinkEvent(0, 1, 0, 400, 0.05)], seed=1)
        sim, m = run_episode(sc, ShortestPathRouter(), G)
        self.assertGreater(m["dropped_overflow"], 0)
        for link in sim.link_list:
            self.assertLessEqual(len(link.queue), link.cap)

    def test_shortest_path_is_static(self):
        sim, _ = run_episode(named_scenario("heavy", seed=2, duration=500), ShortestPathRouter(), G)
        for flow, paths in sim.flow_paths.items():
            self.assertEqual(len(paths), 1, f"flow {flow} used {len(paths)} paths")

    def test_congestion_aware_reroutes(self):
        sim, _ = run_episode(named_scenario("heavy", seed=2, duration=800), CongestionAwareRouter(), G)
        self.assertGreater(len(sim.flow_paths[0]), 1)


class TestDQNComponents(unittest.TestCase):
    def test_gradient_check(self):
        """Hand-written backprop matches finite differences."""
        rng = np.random.default_rng(0)
        net = NumpyQNetwork(6, 3, hidden=8, seed=1)
        net.params = [p.astype(np.float64) for p in net.params]
        x = rng.standard_normal((5, 6))
        a = rng.integers(0, 3, 5)
        y = rng.standard_normal(5) * 0.3
        _, grads = net.gradients(x, a, y)
        eps = 1e-6
        for pi in range(len(net.params)):
            p = net.params[pi]
            for _ in range(5):
                idx = tuple(rng.integers(0, s) for s in p.shape)
                old = p[idx]
                p[idx] = old + eps
                lp, _ = net.gradients(x, a, y)
                p[idx] = old - eps
                lm, _ = net.gradients(x, a, y)
                p[idx] = old
                num = (lp - lm) / (2 * eps)
                self.assertAlmostEqual(num, grads[pi][idx], delta=1e-5 + 1e-3 * abs(num))

    def test_network_learns_regression(self):
        rng = np.random.default_rng(0)
        net = NumpyQNetwork(4, 2, hidden=32, lr=3e-3, seed=0)
        x = rng.standard_normal((256, 4)).astype(np.float32)
        a = rng.integers(0, 2, 256)
        y = (x[:, 0] - x[:, 1] * a).astype(np.float32)
        first = net.train_step(x, a, y)
        for _ in range(400):
            last = net.train_step(x, a, y)
        self.assertLess(last, first * 0.2)

    def test_replay_wraparound(self):
        buf = ReplayBuffer(5, 3, 2)
        for i in range(8):
            buf.add(np.full(3, i), i % 2, float(i), None, None, True)
        self.assertEqual(len(buf), 5)
        self.assertEqual(set(buf.r.tolist()), {3.0, 4.0, 5.0, 6.0, 7.0})

    def test_action_mask(self):
        sc = Scenario("m", BASE_FLOWS, duration=10)
        sim = Simulator(G, sc, ShortestPathRouter())
        enc = StateEncoder(sim)
        enc.refresh(sim)
        from qosroute.simulator import Packet
        p = Packet(0, 0, 0, 9, 1000, 0)
        p.node = 2
        p.visited = (1 << 0) | (1 << 1) | (1 << 2)   # came 0 -> 1 -> 2
        p.hops = 2
        s, mask, nb, _ = enc.encode([p])
        self.assertEqual(s.shape, (1, enc.state_dim))
        allowed = set(nb[0][mask[0]].tolist())
        self.assertEqual(allowed, {3, 5})             # 0 and 1 already visited


class TestMetrics(unittest.TestCase):
    def test_jitter(self):
        self.assertAlmostEqual(jitter({0: [10, 12, 11, 15]}), (2 + 1 + 4) / 3)
        self.assertEqual(jitter({0: [5]}), 0.0)


@unittest.skipUnless(HAS_MODEL, "no trained model yet (run scripts/train.py)")
class TestTrainedModel(unittest.TestCase):
    def test_dqn_paths_mostly_loop_free(self):
        sim, m = run_episode(named_scenario("heavy", seed=4, duration=800), make_router("DQN"), G)
        loops = sum(1 for (_, _, _, path, _) in sim.delivered_log if len(set(path)) != len(path))
        self.assertLessEqual(loops, 0.01 * max(len(sim.delivered_log), 1))
        self.assertGreater(m["pdr"], 0.5)


if __name__ == "__main__":
    unittest.main()
