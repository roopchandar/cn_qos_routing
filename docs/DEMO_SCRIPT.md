# Live demonstration script (≈ 8 minutes)

## Before the review (night before / 15 min before)

```bash
cd drl-qos-routing
source .venv/bin/activate
python -m pytest -q                      # all tests pass
python scripts/demo_congestion.py        # terminal demo works (backup if the browser fails)
streamlit run dashboard/app.py           # leave this tab open
```

Keep `results/figures/` open in Finder as a backup, and the slide deck open.

## 1. The problem (1 min) — dashboard, "Live demo" tab

* Point at the topology: 10 routers, link labels show bandwidth/propagation delay.
* "The shortest path from R0 to R9 is R0–R1–R4–R7–R9. Several flows share that core."

## 2. Inject congestion (2 min)

Sidebar defaults already set this up: demo flow R0→R9 at 4 Mbps, background load 0.6, link
R4–R7 degraded to 20 % bandwidth between 700 and 1400 ms.

1. Move the **"Show network at time"** slider to ~200 ms (before the event): Shortest Path sends
   100 % of R0→R9 through R1–R4–R7 (and is already queueing because other flows share it); the
   adaptive methods split traffic between R1–R4–R7 and R2–R5–R7.
2. Move it to ~900 ms (during the event):
   * **Shortest Path** still sends 100 % through R4–R7 → queue turns red; demo-flow delay
     ≈ 155 ms and only ~68 of ~350 packets delivered in that window (seed 7).
   * **Congestion-Aware** and **DQN** move traffic to R4–R5–R7, R2–R5–R7 and R5–R8 paths
     (demo-flow delay ≈ 23 ms and ≈ 17 ms).
3. Move to ~1600 ms: adaptive algorithms return to the short path once the link recovers.
4. Scroll to the time-series charts: the delay spike and drops for Shortest Path during the
   shaded window; compare DQN vs heuristic delay and jitter.

Say: *"The DQN was never told about this specific event. It learned from many random scenarios to
read queue occupancy and downstream delay and avoid filling queues."*

## 3. Change conditions live (1–2 min)

* Raise **Traffic intensity** to 1.2 → shortest path loses a large fraction of packets even
  without the degraded link.
* Pick another link (e.g. R1–R4) or another source/destination (e.g. R3→R9) to show it is not
  scripted.

## 4. Offline results (2 min) — "Experiment results" tab

* Delay vs load, loss vs load, throughput vs load, PDR vs load: 5 test seeds, mean ± std.
* State honestly where the heuristic is as good as or better than DQN.
* Dynamic scenario time series.

## 5. Training (1 min) — "DQN training" tab

* Reward per packet rises as ε decreases; validation score used to pick the saved model.

## If something fails

* Dashboard won't start → run `python scripts/demo_congestion.py` in the terminal: it prints the
  path shares before/during/after the event for each algorithm.
* Model missing → `python scripts/train.py --episodes 40` (~3 min) gives a usable (weaker) model.
