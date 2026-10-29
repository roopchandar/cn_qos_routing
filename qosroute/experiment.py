"""Glue code shared by the training, evaluation and dashboard scripts."""
from __future__ import annotations

from pathlib import Path

from .metrics import compute_metrics
from .routing import CongestionAwareRouter, DQNRouter, ShortestPathRouter
from .simulator import Simulator
from .topology import build_topology

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "dqn_router.npz"
RESULTS_DIR = PROJECT_ROOT / "results"
ALGORITHMS = ["Shortest Path", "Congestion-Aware", "DQN"]

# Seed ranges are disjoint so training, model selection and final evaluation
# never see the same random traffic.
TRAIN_SEED_BASE = 0            # training episodes use an rng seeded from here
VALIDATION_SEEDS = [500_001, 500_002]
TEST_SEED_BASE = 900_000       # final evaluation seeds: 900000, 900001, ...

_agent_cache: dict = {}


def load_agent(path=DEFAULT_MODEL, backend: str | None = None):
    from .agent.dqn_agent import DQNAgent

    key = (str(path), backend)
    if key not in _agent_cache:
        if not Path(path).exists():
            raise FileNotFoundError(f"No trained model at {path}. Run: python scripts/train.py")
        _agent_cache[key] = DQNAgent.load(path, backend=backend)
    return _agent_cache[key]


def make_router(name: str, model_path=DEFAULT_MODEL, backend: str | None = None, agent=None,
                heuristic_interval: int = 10):
    if name == "Shortest Path":
        return ShortestPathRouter()
    if name == "Congestion-Aware":
        return CongestionAwareRouter(update_interval=heuristic_interval)
    if name == "DQN":
        return DQNRouter(agent if agent is not None else load_agent(model_path, backend), train=False)
    raise ValueError(f"unknown algorithm {name!r}")


def run_episode(scenario, router, graph=None, snapshot_every: int = 0):
    graph = graph if graph is not None else build_topology()
    sim = Simulator(graph, scenario, router, snapshot_every=snapshot_every).run()
    return sim, compute_metrics(sim)
