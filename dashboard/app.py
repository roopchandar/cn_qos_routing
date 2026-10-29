"""Interactive dashboard.  Run from the project root:

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from qosroute.demo import DemoParams, run_demo  # noqa: E402
from qosroute.experiment import ALGORITHMS, DEFAULT_MODEL, RESULTS_DIR, make_router, run_episode  # noqa: E402
from qosroute.metrics import METRIC_LABELS  # noqa: E402
from qosroute.plotting import COLORS, draw_network, plot_training_curve  # noqa: E402
from qosroute.topology import build_topology  # noqa: E402
from qosroute.traffic import load_sweep_scenario  # noqa: E402

st.set_page_config(page_title="DRL QoS Routing", page_icon="🛰️", layout="wide")


# ---------------------------------------------------------------- cached work
@st.cache_resource
def get_graph():
    return build_topology()


@st.cache_data(show_spinner=False)
def cached_demo(params_key, algorithm):
    return run_demo(DemoParams(**dict(params_key)), algorithm, get_graph())


@st.cache_data(show_spinner=False)
def cached_sweep(loads, seeds, algorithms):
    rows = []
    for load in loads:
        for seed in range(seeds):
            for algo in algorithms:
                _, m = run_episode(load_sweep_scenario(load, 990_000 + seed, 1500), make_router(algo), get_graph())
                rows.append({"load": load, "algorithm": algo, **m})
    return pd.DataFrame(rows)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


G = get_graph()
MODEL_OK = DEFAULT_MODEL.exists()
NODES = sorted(G.nodes)
EDGES = sorted(tuple(sorted(e)) for e in G.edges)

# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.title("Scenario controls")
algos_avail = ALGORITHMS if MODEL_OK else ALGORITHMS[:2]
if not MODEL_OK:
    sb.warning("No trained DQN model found. Run `python scripts/train.py` first. "
               "Showing baselines only.")
algos = sb.multiselect("Routing algorithms", algos_avail, default=algos_avail)

sb.subheader("Tracked demo flow")
c1, c2 = sb.columns(2)
src = c1.selectbox("Source", NODES, index=0, format_func=lambda n: f"R{n}")
dst = c2.selectbox("Destination", NODES, index=9, format_func=lambda n: f"R{n}")
demo_rate = sb.slider("Demo flow rate (Mbps)", 0.5, 8.0, 4.0, 0.5)

sb.subheader("Background traffic")
background = sb.checkbox("Hotspot background traffic", value=True)
load = sb.slider("Traffic intensity (load multiplier)", 0.1, 2.0, 0.6, 0.1,
                 help="1.0 ≈ 23 Mbps offered by the 7 background flows")

sb.subheader("Congestion injection")
inject = sb.checkbox("Degrade a link", value=True)
link_idx = EDGES.index((4, 7)) if (4, 7) in EDGES else 0
cong_link = sb.selectbox("Link", EDGES, index=link_idx, format_func=lambda e: f"R{e[0]} – R{e[1]}")
cong_factor = sb.slider("Remaining bandwidth during event", 0.05, 1.0, 0.2, 0.05)
duration = sb.select_slider("Simulated duration (ms)", options=[1000, 1500, 2000, 3000], value=2000)
cong_window = sb.slider("Event window (ms)", 0, duration, (int(duration * 0.35), int(duration * 0.7)), 50)
seed = int(sb.number_input("Random seed", min_value=0, max_value=10_000, value=7, step=1))

params = DemoParams(src=src, dst=dst, demo_rate_mbps=demo_rate, background=background, load=load,
                    cong_link=cong_link if inject else None, cong_factor=cong_factor,
                    cong_start=cong_window[0], cong_end=cong_window[1], duration=duration, seed=seed)

st.title("Deep RL-Based Adaptive Congestion-Aware Routing")
st.caption("Packet-level discrete-time simulation (1 tick = 1 ms). All algorithms see identical "
           "packet arrivals for the same seed. Results are simulated, not measured on real hardware.")

tab_demo, tab_results, tab_train, tab_about = st.tabs(
    ["Live demo", "Experiment results", "DQN training", "How it works"])

# ================================================================ LIVE DEMO
with tab_demo:
    if src == dst:
        st.error("Source and destination must differ.")
    elif not algos:
        st.info("Select at least one routing algorithm in the sidebar.")
    else:
        with st.spinner("Simulating ..."):
            results = {a: cached_demo(params.key(), a) for a in algos}

        # ---- KPI table
        rows = []
        for a, r in results.items():
            m, f = r["metrics"], r["demo_flow"]
            rows.append({"Algorithm": a,
                         "Avg delay (ms)": m["avg_delay_ms"], "P95 delay (ms)": m["p95_delay_ms"],
                         "Jitter (ms)": m["jitter_ms"], "Throughput (Mbps)": m["throughput_mbps"],
                         "PDR": m["pdr"], "Loss ratio": m["loss_ratio"],
                         "Congestion freq": m["congestion_freq"],
                         "Demo flow delay (ms)": f["avg_delay_ms"], "Demo flow PDR": f["pdr"]})
        kpi = pd.DataFrame(rows).set_index("Algorithm")
        st.subheader("QoS summary (whole network + tracked demo flow)")
        st.dataframe(kpi.round(3))

        cols = st.columns(len(results))
        for col, (a, r) in zip(cols, results.items()):
            m = r["metrics"]
            col.metric(f"{a}: avg delay", f"{m['avg_delay_ms']:.1f} ms")
            col.metric(f"{a}: delivery ratio", f"{m['pdr'] * 100:.1f} %")

        # ---- topology at a chosen time
        st.subheader("Network state and routes of the demo flow")
        t_sel = st.slider("Show network at time (ms)", 0, duration - 25, min(cong_window[0] + 100, duration - 25), 25)
        cols = st.columns(len(results))
        for col, (a, r) in zip(cols, results.items()):
            snap = min(r["snapshots"], key=lambda s: abs(s["t"] - t_sel))
            period = next((name for name, p in r["periods"].items()
                           if p["window"][0] <= t_sel < p["window"][1]), list(r["periods"])[-1])
            paths = [(p, n) for p, n, _ in r["periods"][period]["paths"][:4]]
            in_event = inject and cong_window[0] <= t_sel < cong_window[1]
            with col:
                fig, ax = plt.subplots(figsize=(6.4, 3.9))
                draw_network(G, snap["occupancy"], paths, ax=ax, endpoints=(src, dst),
                             degraded={cong_link} if in_event else None,
                             title=f"{a}  ·  t = {snap['t']} ms  ·  routes '{period}'")
                show_fig(fig)
                with st.expander(f"{a}: demo-flow paths per period", expanded=False):
                    for name, p in r["periods"].items():
                        st.markdown(f"**{name}** ({p['window'][0]}–{p['window'][1]} ms) — "
                                    f"avg delay {p['avg_delay_ms']:.1f} ms, delivered {p['delivered']}")
                        st.table(pd.DataFrame([{"path": " → ".join(f"R{x}" for x in path),
                                                "share": f"{s * 100:.1f}%"} for path, _, s in p["paths"][:5]]))
        st.caption("Link colour = queue occupancy at the selected time (green empty → red full). "
                   "Line width = bandwidth. Blue arrows = paths taken by the demo flow in that period "
                   "(thicker = more packets). Dashed = degraded link.")

        # ---- time series
        st.subheader("How QoS evolves over time")
        t_axis = results[algos[0]]["timeseries"]["t"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Average delay of delivered packets (ms), all flows**")
            st.line_chart(pd.DataFrame({a: r["timeseries"]["avg_delay"] for a, r in results.items()}, index=t_axis))
            st.markdown("**Packets dropped per 25 ms**")
            st.line_chart(pd.DataFrame({a: r["timeseries"]["drops"] for a, r in results.items()}, index=t_axis))
        with c2:
            st.markdown(f"**Demo flow R{src}→R{dst} delay (ms)**")
            st.line_chart(pd.DataFrame({a: r["timeseries"]["demo_delay"] for a, r in results.items()}, index=t_axis))
            st.markdown("**Packets waiting in queues (whole network)**")
            st.line_chart(pd.DataFrame({a: r["timeseries"]["queued"] for a, r in results.items()}, index=t_axis))
        if inject:
            st.caption(f"Link R{cong_link[0]}–R{cong_link[1]} runs at {cong_factor:.0%} bandwidth "
                       f"between {cong_window[0]} and {cong_window[1]} ms.")

# ================================================================ RESULTS
with tab_results:
    st.subheader("Offline evaluation (scripts/evaluate.py)")
    figdir = RESULTS_DIR / "figures"
    figs = [("delay_vs_load.png", "Average delay vs load"), ("loss_vs_load.png", "Packet loss vs load"),
            ("throughput_vs_load.png", "Throughput vs load"), ("pdr_vs_load.png", "Delivery ratio vs load"),
            ("jitter_vs_load.png", "Jitter vs load"), ("congestion_vs_load.png", "Congestion frequency vs load")]
    if (figdir / figs[0][0]).exists():
        for i in range(0, len(figs), 2):
            cols = st.columns(2)
            for col, (fname, cap) in zip(cols, figs[i:i + 2]):
                if (figdir / fname).exists():
                    col.image(str(figdir / fname), caption=cap)
        for fname in ("scenario_comparison.png", "dynamic_timeseries.png"):
            if (figdir / fname).exists():
                st.image(str(figdir / fname))
        md = RESULTS_DIR / "RESULTS.md"
        if md.exists():
            with st.expander("All result tables (mean ± std over seeds)"):
                st.markdown(md.read_text())
    else:
        st.info("No evaluation results yet. Run `python scripts/evaluate.py`.")

    st.subheader("Quick live load sweep")
    st.caption("Runs now on fresh seeds (1.5 s of traffic each) — smaller than the offline evaluation.")
    c1, c2 = st.columns(2)
    sweep_loads = c1.multiselect("Loads", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                                 default=[0.5, 1.0, 1.5, 2.0])
    sweep_seeds = c2.slider("Seeds per point", 1, 5, 2)
    metric = st.selectbox("Metric", ["avg_delay_ms", "loss_ratio", "throughput_mbps", "pdr", "jitter_ms",
                                     "congestion_freq"], format_func=lambda k: METRIC_LABELS[k])
    if st.button("Run sweep") and sweep_loads and algos:
        with st.spinner("Running sweep ..."):
            df = cached_sweep(tuple(sorted(sweep_loads)), sweep_seeds, tuple(algos))
        piv = df.groupby(["load", "algorithm"])[metric].mean().unstack()
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        for a in piv.columns:
            ax.plot(piv.index, piv[a], marker="o", color=COLORS[a], label=a)
        ax.set_xlabel("Load multiplier")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        show_fig(fig)
        st.dataframe(piv)

# ================================================================ TRAINING
with tab_train:
    log = RESULTS_DIR / "training_log.csv"
    if log.exists():
        show_fig(plot_training_curve(log))
        df = pd.read_csv(log)
        c1, c2, c3 = st.columns(3)
        c1.metric("Episodes", len(df))
        c2.metric("Final epsilon", f"{df.epsilon.iloc[-1]:.3f}")
        c3.metric("Last-10 reward / packet", f"{df.reward_per_packet.tail(10).mean():.2f}")
        if MODEL_OK:
            meta = json.loads(str(np.load(DEFAULT_MODEL)["meta"]))
            st.markdown(f"**Saved model:** `models/{DEFAULT_MODEL.name}` — best validation score "
                        f"{meta.get('val_score', float('nan')):.3f} at episode {meta.get('episode')}; "
                        f"state size {meta['state_dim']}, actions {meta['n_actions']}.")
            st.json(meta["config"])
        with st.expander("Training log"):
            st.dataframe(df)
    else:
        st.info("No training log yet. Run `python scripts/train.py`.")

# ================================================================ ABOUT
with tab_about:
    st.markdown((ROOT / "docs" / "HOW_IT_WORKS.md").read_text()
                if (ROOT / "docs" / "HOW_IT_WORKS.md").exists() else "See README.md")
