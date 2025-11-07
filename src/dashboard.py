# dashboard.py
import json
import random
import requests
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from network import build_network

st.set_page_config(page_title="QuantumNet Dashboard", layout="wide")
st.title("QuantumNet Dashboard")

# Nota: il server Flask deve girare su https://localhost:5000
API = "https://localhost:5000"

# ---- grafo locale solo per la visualizzazione ----
G, nodes_local = build_network(num_nodes=4, eve_prob=0.0)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Network Graph")
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos=pos, with_labels=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Link metrics (QBER vs Noise)")
    node_id = st.selectbox("Select Node", list(nodes_local.keys()), format_func=lambda i: nodes_local[i].name)
    try:
        metrics = requests.get(f"{API}/metrics/{node_id}", verify=False).json()
        if "error" in metrics:
            st.error(metrics["error"])
        else:
            for neighbor, data in metrics.items():
                st.markdown(f"**To Node {neighbor}**")
                qber = data.get("qber", [])
                noise = data.get("noise", [])
                if qber and noise:
                    df = {"Noise": noise, "QBER": qber}
                    st.plotly_chart(px.line(df, x="Noise", y="QBER", markers=True, title=f"Node {node_id} → {neighbor}"))
                else:
                    st.info("No metrics yet.")
    except Exception as e:
        st.error(f"Cannot fetch metrics: {e}")

st.divider()
st.header("Confronto protocolli")
st.write("- **BB84**: information-theoretic security; rileva Eve (QBER ↑); richiede hw quantistico.")
st.write("- **E91**: basato su entanglement; robusto con test CHSH; latenza/risorse maggiori.")
st.write("- **Kyber**: post-quantum classico (computational); nessun hw quantistico; diverso modello di sicurezza.")

st.divider()
st.header("Secure Chat (demo)")
st.caption("Per prova manuale da client Socket.IO usare gli eventi `join`, `message`, `decrypt` come da API del server.")

with st.expander("Esempio rapido"):
    st.code("""
# in un client Socket.IO (browser o Python):
socket.emit("join", {"node_id": 0})
socket.emit("join", {"node_id": 1})
socket.emit("message", {"sender": 0, "receiver": 1, "msg": "Ciao da 0→1"})
# al client del receiver arriva 'secure_msg' con blob_hex
# per test decifrare lato server (demo):
socket.emit("decrypt", {"owner": 1, "peer": 0, "blob_hex": "<hex>"})
""", language="python")
