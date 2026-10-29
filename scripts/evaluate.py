"""Evaluate all three routing algorithms on held-out test scenarios.

Usage (from the project root):
    python scripts/evaluate.py              # full evaluation (5 seeds per point)
    python scripts/evaluate.py --seeds 2    # faster

Experiments (all use TEST seeds, disjoint from training/validation seeds):
  1. Load sweep on the hotspot traffic matrix, load multiplier 0.25 ... 2.0
  2. Named scenarios: light, moderate, heavy, dynamic (load steps + link degradation)
  3. Generalisation: 10 random, never-seen traffic matrices at two loads
  4. Sensitivity: congestion-aware heuristic with slower link-state updates

Every algorithm sees exactly the same packet arrivals for a given seed
(common random numbers), so differences come from routing alone.

Outputs: results/eval_runs.csv (every run), results/summary_*.csv,
results/RESULTS.md (tables) and results/figures/*.png.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qosroute.experiment import (ALGORITHMS, RESULTS_DIR, TEST_SEED_BASE, load_agent,  # noqa: E402
                                 make_router, run_episode)
from qosroute.topology import build_topology  # noqa: E402
from qosroute.traffic import Scenario, load_sweep_scenario, named_scenario, random_flows  # noqa: E402
from qosroute.plotting import COLORS, plot_training_curve  # noqa: E402

LOADS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
KEEP = ["generated", "delivered", "dropped_overflow", "dropped_ttl", "in_flight", "offered_mbps",
        "avg_delay_ms", "p95_delay_ms", "jitter_ms", "throughput_mbps", "pdr", "loss_ratio",
        "undelivered_ratio", "avg_link_util", "max_link_util", "congestion_freq", "avg_hops"]


def run_all(experiment, scenario, graph, rows, algos=ALGORITHMS, label_extra=None, **router_kw):
    for algo in algos:
        router = make_router(algo, **router_kw)
        _, m = run_episode(scenario, router, graph)
        row = {"experiment": experiment, "scenario": scenario.name, "seed": scenario.seed,
               "algorithm": algo if label_extra is None else f"{algo} {label_extra}",
               "load": scenario.load if scenario.schedule is None else np.nan}
        row.update({k: m[k] for k in KEEP})
        rows.append(row)


def summarise(df, by):
    metrics = ["avg_delay_ms", "p95_delay_ms", "jitter_ms", "throughput_mbps", "pdr", "loss_ratio",
               "congestion_freq", "max_link_util", "avg_hops"]
    g = df.groupby(by)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    return out.reset_index()


def line_plot(summary, metric, ylabel, fname, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for algo in ALGORITHMS:
        s = summary[summary.algorithm == algo].sort_values("load")
        ax.errorbar(s.load, s[f"{metric}_mean"], yerr=s[f"{metric}_std"], label=algo,
                    color=COLORS[algo], marker="o", ms=4, capsize=3, lw=1.8)
    ax.set_xlabel("Traffic load multiplier (1.0 ≈ 23 Mbps offered)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / fname, dpi=150)
    plt.close(fig)


def dynamic_timeseries(graph, seed):
    """Per-window average delay and drops over time in the dynamic scenario."""
    sc = named_scenario("dynamic", seed=seed)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True)
    win = 50
    for algo in ALGORITHMS:
        sim, _ = run_episode(sc, make_router(algo), graph)
        n = sc.duration + sc.drain
        k = n // win
        dl = sim.ts_delivered[: k * win].reshape(k, win).sum(1)
        ds = sim.ts_delay_sum[: k * win].reshape(k, win).sum(1)
        dr = sim.ts_dropped[: k * win].reshape(k, win).sum(1)
        t = np.arange(k) * win + win / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            axes[0].plot(t, np.where(dl > 0, ds / dl, np.nan), color=COLORS[algo], label=algo, lw=1.6)
        axes[1].plot(t, dr, color=COLORS[algo], label=algo, lw=1.6)
    d = sc.duration
    for ax in axes:
        ax.axvspan(d // 3, 2 * d // 3, color="red", alpha=0.07)
        for x in (d // 4, d // 2, 3 * d // 4):
            ax.axvline(x, color="gray", ls=":", lw=1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Avg delay of packets\ndelivered in window (ms)")
    axes[1].set_ylabel(f"Packets dropped\nper {win} ms")
    axes[1].set_xlabel("Simulation time (ms)   [shaded: link 4–7 at 30 % bandwidth; dotted: load changes]")
    axes[0].set_title("Dynamic scenario: load 0.4 → 1.0 → 1.6 → 0.8 with link degradation")
    axes[0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "dynamic_timeseries.png", dpi=150)
    plt.close(fig)


def scenario_bars(summary):
    names = ["light", "moderate", "heavy", "dynamic"]
    metrics = [("avg_delay_ms", "Avg delay (ms)"), ("loss_ratio", "Packet loss ratio"),
               ("throughput_mbps", "Throughput (Mbps)"), ("jitter_ms", "Jitter (ms)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    x = np.arange(len(names))
    w = 0.26
    for ax, (m, lab) in zip(axes, metrics):
        for i, algo in enumerate(ALGORITHMS):
            s = summary[summary.algorithm == algo].set_index("scenario").loc[names]
            ax.bar(x + (i - 1) * w, s[f"{m}_mean"], w, yerr=s[f"{m}_std"], color=COLORS[algo],
                   label=algo, capsize=2)
        ax.set_xticks(x, names)
        ax.set_title(lab)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "scenario_comparison.png", dpi=150)
    plt.close(fig)


def fmt(mean, std, digits=2):
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_markdown(sweep, named, gen, sens, n_seeds, meta, paired):
    lines = ["# Evaluation results (generated by scripts/evaluate.py)", "",
             f"Seeds per data point: {n_seeds} (test seeds {TEST_SEED_BASE}+). "
             "Values are mean ± standard deviation across seeds. Simulated time unit: 1 ms.",
             f"DQN model: checkpoint from training episode {meta.get('episode', '?')} "
             f"(selected by best validation score {meta.get('val_score', float('nan')):.3f}).", ""]

    def table(df, key, keys, algos=ALGORITHMS):
        hdr = (f"| {key} | Algorithm | Avg delay (ms) | Jitter (ms) | Throughput (Mbps) | PDR | Loss ratio "
               "| Congestion freq | Avg hops |")
        out = [hdr, "|" + "---|" * 9]
        for k in keys:
            for algo in algos:
                r = df[(df[key] == k) & (df.algorithm == algo)]
                if r.empty:
                    continue
                r = r.iloc[0]
                out.append(f"| {k} | {algo} | {fmt(r.avg_delay_ms_mean, r.avg_delay_ms_std, 1)} | "
                           f"{fmt(r.jitter_ms_mean, r.jitter_ms_std, 2)} | "
                           f"{fmt(r.throughput_mbps_mean, r.throughput_mbps_std, 2)} | "
                           f"{fmt(r.pdr_mean, r.pdr_std, 3)} | {fmt(r.loss_ratio_mean, r.loss_ratio_std, 3)} | "
                           f"{fmt(r.congestion_freq_mean, r.congestion_freq_std, 3)} | "
                           f"{r.avg_hops_mean:.2f} |")
        return out

    lines += ["## 1. Load sweep (hotspot traffic matrix)", ""]
    lines += table(sweep, "load", LOADS) + [""]
    lines += ["## 2. Named scenarios", ""]
    lines += table(named, "scenario", ["light", "moderate", "heavy", "dynamic"]) + [""]
    lines += ["## 3. Generalisation to unseen random traffic matrices", "",
              "10 random traffic matrices (4–8 flows, 1–5 Mbps each) never used in training.", ""]
    lines += table(gen, "load", sorted(gen.load.unique())) + [""]
    lines += ["Paired comparison per traffic matrix (same matrix, same seed):", ""]
    for load, (wins_d, wins_p, n) in paired.items():
        lines.append(f"* load {load}: DQN had lower average delay than Congestion-Aware in {wins_d}/{n} "
                     f"matrices and PDR at least as high in {wins_p}/{n}.")
    lines.append("")
    lines += ["## 4. Sensitivity: heuristic link-state update interval", "",
              "Congestion-aware heuristic recomputing routes every 10 / 50 / 100 ms "
              "(DQN decides every hop from current local state).", ""]
    sens_algos = ALGORITHMS[:2] + ["Congestion-Aware (50 ms)", "Congestion-Aware (100 ms)", "DQN"]
    lines += table(sens, "scenario", ["heavy", "dynamic"], sens_algos) + [""]
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--duration", type=int, default=2000)
    args = ap.parse_args()
    seeds = [TEST_SEED_BASE + i for i in range(args.seeds)]
    graph = build_topology()
    agent = load_agent()
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []

    print("1/4 load sweep ...", flush=True)
    for load in LOADS:
        for seed in seeds:
            run_all("load_sweep", load_sweep_scenario(load, seed, args.duration), graph, rows)
    print("2/4 named scenarios ...", flush=True)
    for name in ("light", "moderate", "heavy", "dynamic"):
        for seed in seeds:
            run_all("named", named_scenario(name, seed, args.duration), graph, rows)
    print("3/4 unseen random traffic matrices ...", flush=True)
    rng = np.random.default_rng(424242)
    matrices = [random_flows(rng, graph.number_of_nodes()) for _ in range(10)]
    for load in (0.8, 1.4):
        for i, flows in enumerate(matrices):
            sc = Scenario(f"random{i}", flows, duration=args.duration, load=load, seed=TEST_SEED_BASE + 50 + i)
            run_all("generalisation", sc, graph, rows)
    print("4/4 heuristic update-interval sensitivity ...", flush=True)
    for name in ("heavy", "dynamic"):
        for seed in seeds:
            sc = named_scenario(name, seed, args.duration)
            for iv in (50, 100):
                run_all("sensitivity", sc, graph, rows, algos=["Congestion-Aware"],
                        label_extra=f"({iv} ms)", heuristic_interval=iv)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "eval_runs.csv", index=False)
    sweep = summarise(df[df.experiment == "load_sweep"], ["load", "algorithm"])
    named = summarise(df[df.experiment == "named"], ["scenario", "algorithm"])
    gen = summarise(df[df.experiment == "generalisation"], ["load", "algorithm"])
    sens_df = pd.concat([df[(df.experiment == "named") & df.scenario.isin(["heavy", "dynamic"])],
                         df[df.experiment == "sensitivity"]])
    sens = summarise(sens_df, ["scenario", "algorithm"])
    sweep.to_csv(RESULTS_DIR / "summary_load_sweep.csv", index=False)
    named.to_csv(RESULTS_DIR / "summary_scenarios.csv", index=False)
    gen.to_csv(RESULTS_DIR / "summary_generalisation.csv", index=False)
    sens.to_csv(RESULTS_DIR / "summary_sensitivity.csv", index=False)

    line_plot(sweep, "avg_delay_ms", "Average end-to-end delay (ms)", "delay_vs_load.png",
              "Average delay vs traffic load")
    line_plot(sweep, "loss_ratio", "Packet loss ratio", "loss_vs_load.png", "Packet loss vs traffic load")
    line_plot(sweep, "throughput_mbps", "Throughput (Mbps)", "throughput_vs_load.png",
              "Throughput vs traffic load")
    line_plot(sweep, "pdr", "Packet delivery ratio", "pdr_vs_load.png", "Delivery ratio vs traffic load")
    line_plot(sweep, "jitter_ms", "Jitter (ms)", "jitter_vs_load.png", "Jitter vs traffic load")
    line_plot(sweep, "congestion_freq", "Fraction of (link, ms) with queue ≥ 80 %",
              "congestion_vs_load.png", "Congestion frequency vs traffic load")
    scenario_bars(named)
    dynamic_timeseries(graph, seeds[0])
    plot_training_curve(RESULTS_DIR / "training_log.csv", RESULTS_DIR / "figures" / "training_reward.png")
    g = df[df.experiment == "generalisation"]
    paired = {}
    for load in sorted(g.load.unique()):
        a = g[(g.load == load) & (g.algorithm == "DQN")].set_index("scenario")
        b = g[(g.load == load) & (g.algorithm == "Congestion-Aware")].set_index("scenario")
        paired[load] = (int((a.avg_delay_ms < b.avg_delay_ms).sum()), int((a.pdr >= b.pdr).sum()), len(a))
    write_markdown(sweep, named, gen, sens, args.seeds, getattr(agent, "meta", {}), paired)
    print(f"Done in {time.time() - t0:.0f}s. See results/RESULTS.md and results/figures/")


if __name__ == "__main__":
    main()
