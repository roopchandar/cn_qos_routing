# Deep Reinforcement Learning-Based Adaptive Congestion-Aware Routing for QoS-Aware Computer Networks

A packet-level network simulator in which a **Deep Q-Network (DQN)** learns to choose the next hop of
every packet from the current network state, compared against **static shortest-path routing** and a
**congestion-aware heuristic** under light, moderate, heavy and dynamically changing traffic.

> "Congestion-free" is the *objective*: the agent tries to minimise congestion, delay and loss.
> No routing scheme can guarantee zero congestion when offered traffic exceeds network capacity.

![architecture](docs/figures/architecture.png)

---

## 1. Quick start on macOS (Apple Silicon, VS Code)

```bash
# 0) Python 3.10 – 3.12 recommended (3.13 also works). If you need one:
#    brew install python@3.12
cd drl-qos-routing

# 1) virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) core dependencies (all have native arm64 wheels)
pip install -r requirements.txt

# 3) OPTIONAL: PyTorch backend for the Q-network (native Apple Silicon wheel from PyPI)
pip install torch

# 4) check everything works (≈ 30 s)
python -m pytest -q            # or: python -m unittest discover -s tests
```

In VS Code: *Cmd+Shift+P → Python: Select Interpreter →* `.venv`.

### Run the project

| What | Command | Time (approx.) |
|---|---|---|
| Interactive dashboard | `streamlit run dashboard/app.py` | opens browser |
| Congestion demo in the terminal | `python scripts/demo_congestion.py` | ~5 s |
| Re-train the DQN (optional, a trained model is included) | `python scripts/train.py` | ~15–20 min |
| Quick training run | `python scripts/train.py --episodes 40` | ~3 min |
| Train with PyTorch instead of NumPy | `python scripts/train.py --backend torch` | similar |
| Full evaluation + all graphs | `python scripts/evaluate.py` | ~2–3 min |
| Regenerate diagrams | `python scripts/make_diagrams.py` | seconds |

**Note:** retraining overwrites `models/dqn_router.npz` and `results/training_log.csv`; re-run
`scripts/evaluate.py` afterwards so the graphs match the new model. Back up `models/` first if you
want to keep the included model.

---

## 2. Project structure

```
drl-qos-routing/
├── qosroute/                    # the Python package
│   ├── topology.py              # Module 1: routers, links (bandwidth, delay, queue size)
│   ├── traffic.py               # Module 2: flows, Poisson packets, load schedules, link events
│   ├── simulator.py             # discrete-time packet simulator (1 tick = 1 ms)
│   ├── metrics.py               # Module 5: QoS metric definitions
│   ├── routing/
│   │   ├── base.py              # router interface + Dijkstra next-hop tables
│   │   ├── shortest_path.py     # Module 3: static shortest path (fixed costs)
│   │   ├── congestion_aware.py  # Module 3: dynamic-cost heuristic
│   │   └── dqn_router.py        # Module 4: per-hop DQN routing + reward
│   ├── agent/
│   │   ├── features.py          # state vector + action mask
│   │   ├── qnet_numpy.py        # Q-network in NumPy (default, hand-written backprop + Adam)
│   │   ├── qnet_torch.py        # same network in PyTorch (optional backend)
│   │   ├── replay.py            # experience replay buffer
│   │   └── dqn_agent.py         # DQN: target net, ε-greedy, Double-DQN update, save/load
│   ├── experiment.py            # helpers: make_router, run_episode, seeds
│   ├── demo.py                  # congestion-injection demo logic
│   └── plotting.py              # network drawing + training curve
├── scripts/
│   ├── train.py                 # train the DQN, save best model
│   ├── evaluate.py              # all experiments → CSV, tables, graphs
│   ├── demo_congestion.py       # terminal demo
│   └── make_diagrams.py         # architecture / flowchart / topology PNGs
├── dashboard/app.py             # Module 6: Streamlit dashboard
├── tests/                       # unit tests (+ dashboard smoke test, torch-equivalence test)
├── models/dqn_router.npz        # trained model (weights + config)
├── results/                     # training_log.csv, eval_runs.csv, summaries, RESULTS.md, figures/
└── docs/                        # report, viva Q&A, how-it-works, demo script, diagrams
```

---

## 3. What is simulated

* **Topology:** 10 routers, 15 full-duplex links (5–10 Mbps, 1–3 ms propagation), 30-packet
  FIFO output queue per link direction. Several alternative paths exist between busy pairs.
* **Traffic:** flows with Poisson packet arrivals, packet size uniform 500–1500 B. The evaluation
  "hotspot" matrix has 7 flows; load multiplier 1.0 ≈ 23 Mbps offered.
* **Per packet:** queueing delay, transmission (byte-credit scheduler), propagation delay,
  tail-drop when a queue is full, TTL = 12 hops.
* **Dynamic conditions:** load steps and link degradation (a link runs at e.g. 20–30 % bandwidth).

Simulated values are *model outputs*, not measurements from real routers. See
[docs/REPORT.md](docs/REPORT.md) §8 for limitations and how to move to Mininet/SDN.

## 4. The three routing algorithms

| Algorithm | Decision | Information used |
|---|---|---|
| Shortest Path | fixed next-hop table (Dijkstra, cost = propagation delay) | topology only |
| Congestion-Aware | Dijkstra on cost = prop + queue delay + tx time (+ penalty if ≥ 90 % full), recomputed every 10 ms | global, refreshed periodically |
| DQN | per hop, argmax of learned Q-values over allowed neighbours | current local link state + one-hop downstream look-ahead |

## 5. DQN design (summary)

* **State (65 values):** for each of up to 4 neighbours — queue occupancy, queue delay, propagation
  delay, bandwidth, hops & delay from that neighbour to the destination, downstream queue delay,
  is-destination flag, recent drops, recent utilisation — plus one-hot current router, one-hot
  destination and remaining TTL.
* **Action:** which neighbour to forward to (4 slots). **Action mask** removes non-existent
  neighbours, already-visited routers (prevents loops) and neighbours from which the destination
  cannot be reached within the remaining TTL.
* **Reward:** `r = 10·delivered − 0.1·hop_delay_ms − 5·dropped − 3·congestion_penalty`,
  where congestion_penalty = max(0, occupancy − 0.7)/0.3 of the chosen queue.
* **Learning:** experience replay (100k), Double-DQN target with soft-updated target network
  (τ = 0.01), Huber loss, Adam (lr 5e-4), γ = 0.98, ε-greedy 1.0 → 0.05.
* **Training:** 300 episodes of randomly generated scenarios; model selected on separate
  validation seeds; evaluated on separate test seeds and on unseen random traffic matrices.

Full explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## 6. Results

See [results/RESULTS.md](results/RESULTS.md) (generated tables) and
[docs/REPORT.md](docs/REPORT.md) (discussion). Figures are in `results/figures/`.

## 7. Documentation

* [docs/REPORT.md](docs/REPORT.md) — project report
* [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) — concepts explained simply (also shown in the dashboard)
* [docs/VIVA.md](docs/VIVA.md) — viva questions with answers
* [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) — step-by-step script for the live demonstration
* Slide deck — the separate "DRL Adaptive Routing — Project Review" deck (downloadable as PowerPoint/PDF)

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: qosroute` | run commands from the project root folder |
| `No trained model at models/dqn_router.npz` | `python scripts/train.py` (or restore the file) |
| `Model was trained for a different topology` | you changed `topology.py`; retrain |
| `streamlit: command not found` | activate the venv: `source .venv/bin/activate` |
| torch install fails | skip it — the default NumPy backend needs no PyTorch |
| Dashboard port busy | `streamlit run dashboard/app.py --server.port 8502` |
# cn_qos_routing
