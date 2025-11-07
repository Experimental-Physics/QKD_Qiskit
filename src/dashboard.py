# dashboard.py
import json
import requests
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from network import build_network

st.set_page_config(page_title="QuantumNet Dashboard", layout="wide")
st.title("QuantumNet Dashboard")

API = "https://localhost:5000"  # server Flask/Socket.IO
REQUESTS_VERIFY = False          # self-signed cert in demo

# ---------------- Sidebar controls ----------------
st.sidebar.header("Network Controls")

num_nodes = st.sidebar.slider("Number of nodes", min_value=2, max_value=16, value=4, step=1)
noise_min_sb = st.sidebar.number_input("Init noise min", 0.0, 1.0, 0.0, 0.01, format="%.2f")
noise_max_sb = st.sidebar.number_input("Init noise max", 0.0, 1.0, 0.05, 0.01, format="%.2f")
eve_prob_sb  = st.sidebar.slider("Init Eve probability", 0.0, 1.0, 0.0, 0.05)

if st.sidebar.button("Rebuild network on server"):
    try:
        resp = requests.post(f"{API}/rebuild",
                             json={"num_nodes": num_nodes,
                                   "noise_range": [noise_min_sb, noise_max_sb],
                                   "eve_prob": eve_prob_sb},
                             verify=REQUESTS_VERIFY, timeout=30)
        st.sidebar.success(resp.json())
    except Exception as e:
        st.sidebar.error(f"Rebuild failed: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: you can still simulate local graphs below for display only.")

# ---------------- Local viz graph (informational) ----------------
G_local, nodes_local = build_network(num_nodes=num_nodes, eve_prob=0.0)

# ---------------- Tabs ----------------
tab_net, tab_analysis = st.tabs(["🌐 Network & Metrics", "🔬 QBER Analysis"])

with tab_net:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Network Graph (local view)")
        fig, ax = plt.subplots()
        pos = nx.spring_layout(G_local, seed=42)
        nx.draw(G_local, pos=pos, with_labels=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Link metrics (QBER vs Noise) from server")
        try:
            node_id = st.selectbox("Select Node (server)", list(range(num_nodes)), format_func=lambda i: f"Node{i}")
            metrics = requests.get(f"{API}/metrics/{node_id}", verify=REQUESTS_VERIFY, timeout=15).json()
            if "error" in metrics:
                st.error(metrics["error"])
            elif not metrics:
                st.info("No metrics yet for this node.")
            else:
                for neighbor, data in metrics.items():
                    st.markdown(f"**To Node {neighbor}**")
                    qber = data.get("qber", [])
                    noise = data.get("noise", [])
                    if qber and noise:
                        df = {"Noise": noise, "QBER": qber}
                        fig = px.line(df, x="Noise", y="QBER", markers=True, title=f"Node {node_id} → {neighbor}")
                        # add 11% threshold
                        fig.add_hline(y=0.11, line_dash="dash", annotation_text="QBER threshold 11%", annotation_position="top left")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No data for this link.")
        except Exception as e:
            st.error(f"Cannot fetch metrics: {e}")

with tab_analysis:
    st.subheader("Sweep: BB84 QBER vs Noise")

    colA, colB = st.columns(2)
    with colA:
        noise_min = st.number_input("Noise min", 0.0, 1.0, 0.0, 0.01, format="%.3f")
        noise_max = st.number_input("Noise max", 0.0, 1.0, 0.12, 0.01, format="%.3f")
        num_points = st.slider("Number of samples", 5, 101, 31, 2)
    with colB:
        n_bits = st.selectbox("Bits per exchange", [64, 96, 128, 192, 256, 512, 1024], index=4)
        eve_on  = st.checkbox("Simulate Eve (intercept-resend approx.)", value=False)

    if st.button("Run sweep"):
        try:
            params = {
                "noise_min": noise_min,
                "noise_max": noise_max,
                "num": num_points,
                "n_bits": n_bits,
                "eve": int(eve_on),
            }
            r = requests.get(f"{API}/sweep_bb84", params=params, verify=REQUESTS_VERIFY, timeout=120)
            data = r.json()
            if "error" in data:
                st.error(data["error"])
            else:
                xs = data["noise"]
                ys = data["qber"]
                succ = data["success"]

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="QBER"))
                fig.add_hline(y=0.11, line_dash="dash", line_color="red",
                              annotation_text="Threshold 11%", annotation_position="top left")

                # Color background by secure/compromised
                fig.add_shape(type="rect", x0=noise_min, x1=noise_max, y0=0.0, y1=0.11,
                              fillcolor="rgba(0,200,0,0.08)", line_width=0, layer="below")
                fig.add_shape(type="rect", x0=noise_min, x1=noise_max, y0=0.11, y1=max(ys+[0.2]),
                              fillcolor="rgba(200,0,0,0.08)", line_width=0, layer="below")

                fig.update_layout(title="BB84: QBER vs Noise", xaxis_title="Noise", yaxis_title="QBER")
                st.plotly_chart(fig, use_container_width=True)

                # Stats
                secure_count = sum(1 for ok in succ if ok)
                st.markdown(f"**Success (key accepted)**: {secure_count}/{len(succ)} "
                            f"({secure_count/len(succ):.0%})")
                st.caption("A key is considered accepted if QBER ≤ 11% and enough sifted bits were available.")

        except Exception as e:
            st.error(f"Sweep failed: {e}")

    st.info("This analysis runs BB84 on the server for the selected noise range and plots the resulting QBER. "
            "Use it to see how quickly the channel becomes unusable (QBER above ~11%).")
