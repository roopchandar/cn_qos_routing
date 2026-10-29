"""Shared plotting helpers (Matplotlib) for scripts and the dashboard."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .topology import DEFAULT_POS

COLORS = {"Shortest Path": "#8c8c8c", "Congestion-Aware": "#e08a1e", "DQN": "#2563eb"}


def plot_training_curve(log_csv, out_png=None, window: int = 10):
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(df.episode, df.reward_per_packet, color="#93c5fd", lw=1, label="per episode")
    ax.plot(df.episode, df.reward_per_packet.rolling(window, min_periods=1).mean(), color="#1d4ed8",
            lw=2, label=f"{window}-episode moving avg")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Cumulative reward / packet")
    ax.set_title("DQN training: reward per generated packet")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax2 = ax.twinx()
    ax2.plot(df.episode, df.epsilon, color="#9ca3af", ls="--", lw=1)
    ax2.set_ylabel("epsilon (dashed)", color="#6b7280")
    ax = axes[1]
    v = df[df.val_score.astype(str).str.len() > 0].copy()
    v["score"] = v.val_score.astype(str).str.rstrip("*").replace("", np.nan).astype(float)
    ax.plot(v.episode, v.score, marker="o", color="#1d4ed8")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("10×PDR − 0.1×delay")
    ax.set_title("Greedy policy on validation scenarios")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    return fig


def draw_network(graph, link_values: dict | None = None, highlight_paths=None, ax=None,
                 title: str = "", vmax: float = 1.0, label: str = "Queue occupancy",
                 degraded: set | None = None, endpoints=None):
    """Draw the topology. Each undirected link is coloured by the max of its two
    directions' values (e.g. queue occupancy); highlighted paths are drawn on top
    with width proportional to the share of packets using them."""
    pos = DEFAULT_POS
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.6))
    else:
        fig = ax.figure
    cmap = plt.get_cmap("RdYlGn_r")
    edges = list(graph.edges())
    vals = []
    for u, v in edges:
        if link_values:
            vals.append(max(link_values.get((u, v), 0.0), link_values.get((v, u), 0.0)))
        else:
            vals.append(0.0)
    widths = [1.5 + 2.5 * graph[u][v]["bw_mbps"] / 10 for u, v in edges]
    nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=[cmap(min(x / vmax, 1)) for x in vals],
                           width=widths, ax=ax)
    if highlight_paths:
        total = sum(c for _, c in highlight_paths) or 1
        for path, count in highlight_paths:
            pe = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=pe, edge_color="#2563eb", alpha=0.55,
                                   width=1 + 9 * count / total, ax=ax, arrows=True, arrowstyle="-|>",
                                   arrowsize=12, connectionstyle="arc3,rad=0.08")
    if degraded:
        de = [e for e in edges if (e in degraded or (e[1], e[0]) in degraded)]
        nx.draw_networkx_edges(graph, pos, edgelist=de, style="dashed", edge_color="black", width=2.5, ax=ax)
    node_colors = []
    for n in graph.nodes():
        if endpoints and n == endpoints[0]:
            node_colors.append("#16a34a")
        elif endpoints and n == endpoints[1]:
            node_colors.append("#dc2626")
        else:
            node_colors.append("#e5e7eb")
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, edgecolors="#374151", node_size=520, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels={n: f"R{n}" for n in graph.nodes()}, font_size=9, ax=ax)
    elabels = {(u, v): f"{graph[u][v]['bw_mbps']:.0f}M/{graph[u][v]['prop_ms']}ms" for u, v in edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=elabels, font_size=6.5, ax=ax,
                                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, label=label)
    ax.set_title(title)
    ax.axis("off")
    return fig
