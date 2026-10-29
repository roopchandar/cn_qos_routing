"""Generate architecture and flowchart diagrams (PNG) into docs/figures/.

    python scripts/make_diagrams.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from qosroute.plotting import draw_network  # noqa: E402
from qosroute.topology import build_topology  # noqa: E402

OUT = ROOT / "docs" / "figures"


def box(ax, x, y, w, h, text, fc="#eef2ff", ec="#4338ca", fs=9, bold_first=True, pad=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.01",
                                fc=fc, ec=ec, lw=1.4))
    lines = text.split("\n")
    if bold_first:
        ax.text(x + w / 2, y + h - 0.035, lines[0], ha="center", va="top", fontsize=fs + 1, weight="bold")
        ax.text(x + w / 2, y + h - 0.085, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs - 0.5,
                linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, text="", color="#374151", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, lw=1.3,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.015, text, fontsize=7.5, ha="center",
                color=color, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))


def architecture():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("System architecture", fontsize=14, weight="bold")
    box(ax, 0.02, 0.62, 0.22, 0.3, "Module 1: Topology\n10 routers, 15 links\nbandwidth, prop. delay,\n"
        "queue capacity (NetworkX)\ntopology.py")
    box(ax, 0.02, 0.18, 0.22, 0.34, "Module 2: Traffic\nflows (src, dst, Mbps)\nPoisson arrivals,\n"
        "500–1500 B packets,\nload schedules, link\ndegradation events\ntraffic.py")
    box(ax, 0.32, 0.30, 0.3, 0.52, "Network Simulator (1 ms ticks)\n1 apply link events\n"
        "2 router.on_tick()\n3 packets arrive / deliver\n4 generate new packets\n"
        "5 batched next-hop decisions\n   enqueue or drop (tail-drop, TTL)\n6 links transmit (byte credits)\n"
        "   + propagation delay\n7 sample statistics\nsimulator.py", fc="#ecfeff", ec="#0e7490")
    box(ax, 0.7, 0.66, 0.28, 0.27, "Module 3: Routing algorithms\nShortest Path (static Dijkstra)\n"
        "Congestion-Aware (dynamic costs,\nrecomputed every 10 ms)\nDQN Router (per-hop decision)\nrouting/",
        fc="#fff7ed", ec="#c2410c")
    box(ax, 0.7, 0.30, 0.28, 0.3, "Module 4: DQN agent\nstate encoder + action mask\n"
        "Q-network 65→128→128→4\nreplay buffer, target net,\nε-greedy, Double-DQN target\nagent/",
        fc="#f0fdf4", ec="#15803d")
    box(ax, 0.32, 0.03, 0.3, 0.2, "Module 5: QoS metrics\ndelay, jitter, throughput,\nPDR, loss, utilisation,\n"
        "congestion frequency\nmetrics.py", fc="#fdf4ff", ec="#a21caf")
    box(ax, 0.7, 0.03, 0.28, 0.2, "Module 6: Outputs\nevaluate.py → CSV + graphs\nStreamlit dashboard\n"
        "demo_congestion.py", fc="#fefce8", ec="#a16207")
    arrow(ax, (0.24, 0.77), (0.32, 0.72), "graph")
    arrow(ax, (0.24, 0.35), (0.32, 0.45), "packets")
    arrow(ax, (0.62, 0.72), (0.7, 0.78), "packets at router\n+ link state")
    arrow(ax, (0.7, 0.70), (0.62, 0.62), "next hops", rad=0.2)
    arrow(ax, (0.84, 0.66), (0.84, 0.60), "")
    arrow(ax, (0.47, 0.30), (0.47, 0.23), "counters")
    arrow(ax, (0.62, 0.13), (0.7, 0.13), "metrics")
    ax.text(0.84, 0.625, "DQN router queries agent", fontsize=7.5, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", dpi=150)
    plt.close(fig)


def flowchart():
    fig, ax = plt.subplots(figsize=(8.5, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Per-hop DQN routing decision and learning", fontsize=13, weight="bold")
    steps = [
        ("Packet waiting at router u (destination d)", "#e0f2fe"),
        ("Encode state s: per-neighbour queue, delay, bandwidth,\ndistance-to-d, downstream delay, drops "
         "+ one-hot(u), one-hot(d), TTL", "#e0f2fe"),
        ("Action mask: real neighbour, not visited, d reachable within TTL", "#fef3c7"),
        ("Q-network → Q(s, a) for each neighbour; masked actions = −∞", "#dcfce7"),
        ("Choose a: training → ε-greedy;  evaluation → argmax", "#dcfce7"),
        ("Enqueue on link u→a  (queue full → DROP: r = −5 − 3·congestion)", "#fee2e2"),
        ("Wait in queue + transmit + propagate → arrive at v", "#e0f2fe"),
        ("v = d ? DELIVER: r = 10 − 0.1·hop_delay − 3·congestion (done)\n"
         "else r = −0.1·hop_delay − 3·congestion, next state s' at v", "#fef3c7"),
        ("Store (s, a, r, s', mask', done) in replay buffer", "#f3e8ff"),
        ("Every ms: sample 128 transitions, Double-DQN target\ny = r + γ·Q_target(s', argmax Q_online(s'))"
         ", Huber loss, Adam;\nsoft-update target network (τ = 0.01)", "#f3e8ff"),
    ]
    h = 0.065
    gap = 0.028
    y = 0.93
    for i, (t, c) in enumerate(steps):
        box(ax, 0.06, y - h, 0.82, h, t, fc=c, ec="#475569", fs=9, bold_first=False, pad=0.003)
        if i < len(steps) - 1:
            arrow(ax, (0.5, y - h), (0.5, y - h - gap))
        y -= h + gap
    y_from = 0.93 - 7 * (h + gap) - h / 2
    y_to = 0.93 - h / 2
    ax.add_patch(FancyArrowPatch((0.885, y_from), (0.885, y_to), arrowstyle="-|>", mutation_scale=13,
                                 color="#2563eb", lw=1.5, connectionstyle="arc,angleA=0,angleB=0,armA=40,armB=40,rad=8"))
    ax.text(0.975, (y_from + y_to) / 2, "not delivered:\nrepeat at v", fontsize=8, color="#2563eb",
            rotation=90, va="center", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "dqn_flowchart.png", dpi=150)
    plt.close(fig)


def topology():
    g = build_topology()
    fig, ax = plt.subplots(figsize=(9, 5))
    draw_network(g, None, ax=ax, title="Simulated topology (label = bandwidth / propagation delay)")
    fig.tight_layout()
    fig.savefig(OUT / "topology.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    flowchart()
    topology()
    print(f"Diagrams written to {OUT}")
