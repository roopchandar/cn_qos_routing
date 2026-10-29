"""Train the DQN routing agent.

Usage (from the project root):
    python scripts/train.py                       # default: 200 episodes, NumPy backend
    python scripts/train.py --episodes 50         # quick run
    python scripts/train.py --backend torch       # use PyTorch instead of NumPy

Each episode is a fresh random scenario (hotspot or random traffic matrix,
random load 0.3-2.0, optional load step and optional link degradation), so the
agent sees many traffic conditions. Every ``--val-every`` episodes the greedy
policy is scored on held-out validation scenarios and the best model is saved
to models/dqn_router.npz. Per-episode statistics go to results/training_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qosroute.agent.dqn_agent import DQNAgent, DQNConfig  # noqa: E402
from qosroute.agent.features import StateEncoder  # noqa: E402
from qosroute.experiment import (DEFAULT_MODEL, RESULTS_DIR, VALIDATION_SEEDS,  # noqa: E402
                                 run_episode)
from qosroute.metrics import compute_metrics  # noqa: E402
from qosroute.routing import DQNRouter  # noqa: E402
from qosroute.simulator import Simulator  # noqa: E402
from qosroute.topology import build_topology  # noqa: E402
from qosroute.traffic import named_scenario, random_training_scenario  # noqa: E402


def validation_score(agent, graph) -> tuple[float, dict]:
    """Greedy policy on held-out scenarios. Score mirrors the reward:
    10 x PDR - 0.1 x average delay (higher is better)."""
    scores, details = [], {}
    for name in ("moderate", "heavy", "dynamic"):
        for seed in VALIDATION_SEEDS:
            _, m = run_episode(named_scenario(name, seed=seed, duration=1500),
                               DQNRouter(agent, train=False), graph)
            scores.append(10 * m["pdr"] - 0.1 * m["avg_delay_ms"])
            details[f"{name}/{seed}"] = (m["pdr"], m["avg_delay_ms"])
    return float(np.mean(scores)), details


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--duration", type=int, default=1000, help="ms of traffic per episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--out", default=str(DEFAULT_MODEL))
    args = ap.parse_args()

    graph = build_topology()
    links = [(u, v) for u, v in graph.edges]
    rng = np.random.default_rng(args.seed)

    # Build one simulator just to learn the state/action sizes.
    probe = Simulator(graph, named_scenario("light", duration=10), _Dummy())
    enc = StateEncoder(probe)
    ticks_per_ep = args.duration + 150
    cfg = DQNConfig(seed=args.seed, backend=args.backend,
                    eps_decay_steps=int(0.6 * args.episodes * ticks_per_ep))
    agent = DQNAgent(enc.state_dim, enc.K, cfg)
    print(f"State dim {enc.state_dim}, actions {enc.K}, backend {args.backend}, "
          f"episodes {args.episodes}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "training_log.csv"
    fields = ["episode", "load", "flows", "events", "reward", "reward_per_packet", "epsilon",
              "loss", "pdr", "avg_delay_ms", "loss_ratio", "val_score", "seconds"]
    best = -np.inf
    t_start = time.time()
    with open(log_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for ep in range(1, args.episodes + 1):
            t0 = time.time()
            sc = random_training_scenario(rng, graph.number_of_nodes(), links, duration=args.duration)
            router = DQNRouter(agent, train=True)
            n_loss = len(agent.losses)
            sim = Simulator(graph, sc, router, track_paths=False).run()
            m = compute_metrics(sim)
            ep_losses = agent.losses[n_loss:]
            row = {
                "episode": ep,
                "load": round(sc.load, 3),
                "flows": len(sc.flows),
                "events": len(sc.events),
                "reward": round(router.episode_reward, 2),
                "reward_per_packet": round(router.episode_reward / max(sim.generated, 1), 4),
                "epsilon": round(agent.epsilon, 4),
                "loss": round(float(np.mean(ep_losses)), 5) if ep_losses else "",
                "pdr": round(m["pdr"], 4),
                "avg_delay_ms": round(m["avg_delay_ms"], 2),
                "loss_ratio": round(m["loss_ratio"], 4),
                "val_score": "",
                "seconds": round(time.time() - t0, 2),
            }
            if ep % args.val_every == 0 or ep == args.episodes:
                score, _ = validation_score(agent, graph)
                row["val_score"] = round(score, 3)
                if score > best:
                    best = score
                    agent.save(args.out, meta={"episode": ep, "val_score": score,
                                               "topology_edges": [list(e) for e in links]})
                    row["val_score"] = f"{score:.3f}*"
            w.writerow(row)
            fh.flush()
            print(f"ep {ep:4d} | load {sc.load:4.2f} | R/pkt {row['reward_per_packet']:7.3f} | "
                  f"eps {row['epsilon']:.3f} | PDR {m['pdr']:.3f} | delay {m['avg_delay_ms']:6.1f} ms"
                  f" | val {row['val_score']} | {row['seconds']}s", flush=True)

    print(f"\nDone in {time.time() - t_start:.0f}s. Best validation score {best:.3f}. "
          f"Model: {args.out}  Log: {log_path}")


class _Dummy:
    name = "probe"

    def reset(self, sim):
        pass


if __name__ == "__main__":
    main()
