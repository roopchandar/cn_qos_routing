"""Traffic flows, load schedules, link events and named scenarios.

A *flow* is a (source, destination, rate) triple. Packets of a flow arrive as a
Poisson process: in each 1 ms tick the number of new packets is drawn from a
Poisson distribution whose mean is chosen so that the average offered bit-rate
equals ``rate_mbps x load_multiplier``. Packet sizes are uniform in
[500, 1500] bytes (mean 1000 bytes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

PKT_MIN_BYTES = 500
PKT_MAX_BYTES = 1500
PKT_MEAN_BYTES = (PKT_MIN_BYTES + PKT_MAX_BYTES) / 2


@dataclass(frozen=True)
class Flow:
    src: int
    dst: int
    rate_mbps: float  # rate at load multiplier 1.0

    def packets_per_ms(self, load: float) -> float:
        bytes_per_ms = self.rate_mbps * load * 125.0
        return bytes_per_ms / PKT_MEAN_BYTES


@dataclass
class LinkEvent:
    """Change a link's bandwidth during [start, end). factor=0.3 means the link
    runs at 30 % of its nominal bandwidth (e.g. degradation / interference /
    background traffic not controlled by our routers)."""

    u: int
    v: int
    start: int
    end: int
    factor: float


@dataclass
class Scenario:
    name: str
    flows: list[Flow]
    duration: int = 2000           # ms during which traffic is generated
    drain: int = 300               # extra ms with no new traffic, lets queues empty
    load: float = 1.0              # constant load multiplier (if schedule is None)
    schedule: Callable[[int], float] | None = None  # t -> load multiplier
    events: list[LinkEvent] = field(default_factory=list)
    seed: int = 0

    def load_at(self, t: int) -> float:
        return self.schedule(t) if self.schedule is not None else self.load


# --------------------------------------------------------------------------
# The "hotspot" traffic matrix used for the main evaluation. Several flows
# share the low-delay core path 0-1-4-7-9, so delay-based shortest-path
# routing concentrates load there while side paths stay under-used.
# --------------------------------------------------------------------------
BASE_FLOWS = [
    Flow(0, 9, 6.0),
    Flow(1, 7, 3.0),
    Flow(0, 7, 3.0),
    Flow(3, 9, 3.0),
    Flow(2, 8, 2.0),
    Flow(9, 0, 4.0),
    Flow(6, 4, 2.0),
]

LOAD_LEVELS = {"light": 0.4, "moderate": 0.9, "heavy": 1.6}


class StepSchedule:
    """Piecewise-constant load: list of (start_time, multiplier)."""

    def __init__(self, steps: list[tuple[int, float]]):
        self.steps = sorted(steps)

    def __call__(self, t: int) -> float:
        m = self.steps[0][1]
        for start, mult in self.steps:
            if t >= start:
                m = mult
        return m


def named_scenario(name: str, seed: int = 0, duration: int = 2000) -> Scenario:
    """light / moderate / heavy / dynamic scenarios on the hotspot matrix."""
    if name in LOAD_LEVELS:
        return Scenario(name, BASE_FLOWS, duration=duration, load=LOAD_LEVELS[name], seed=seed)
    if name == "dynamic":
        d = duration
        sched = StepSchedule([(0, 0.4), (d // 4, 1.0), (d // 2, 1.6), (3 * d // 4, 0.8)])
        events = [LinkEvent(4, 7, d // 3, 2 * d // 3, 0.3)]
        return Scenario(name, BASE_FLOWS, duration=d, schedule=sched, events=events, seed=seed)
    raise ValueError(f"unknown scenario {name!r}")


def load_sweep_scenario(load: float, seed: int, duration: int = 2000) -> Scenario:
    return Scenario(f"load={load:.2f}", BASE_FLOWS, duration=duration, load=load, seed=seed)


def random_flows(rng: np.random.Generator, n_nodes: int, n_flows: int | None = None) -> list[Flow]:
    """Random traffic matrix: distinct (src, dst) pairs with random rates."""
    n_flows = int(rng.integers(4, 9)) if n_flows is None else n_flows
    flows, used = [], set()
    while len(flows) < n_flows:
        s, d = (int(x) for x in rng.choice(n_nodes, size=2, replace=False))
        if (s, d) in used:
            continue
        used.add((s, d))
        flows.append(Flow(s, d, float(rng.uniform(1.0, 5.0))))
    return flows


def random_training_scenario(rng: np.random.Generator, n_nodes: int, links: list[tuple[int, int]],
                             duration: int = 1000) -> Scenario:
    """Diverse scenarios for DQN training: hotspot or random matrix, random load,
    optional load steps and optional link degradation."""
    flows = BASE_FLOWS if rng.random() < 0.5 else random_flows(rng, n_nodes)
    load = float(rng.uniform(0.3, 2.0))
    schedule = None
    if rng.random() < 0.3:
        a, b = float(rng.uniform(0.3, 1.0)), float(rng.uniform(0.8, 2.0))
        schedule = StepSchedule([(0, a), (duration // 2, b)])
    events = []
    if rng.random() < 0.5:
        u, v = links[int(rng.integers(len(links)))]
        start = int(rng.integers(0, duration // 2))
        events.append(LinkEvent(u, v, start, start + duration // 2, float(rng.uniform(0.2, 0.6))))
    return Scenario("train", flows, duration=duration, drain=150, load=load, schedule=schedule,
                    events=events, seed=int(rng.integers(1_000_000_000)))
