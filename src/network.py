# network.py
from __future__ import annotations
import random
from typing import Dict, Tuple
import networkx as nx

from qkd import bb84_key_generation, e91_key_generation

class QKDNode:
    def __init__(self, name: str):
        self.name = name
        # neighbor -> (key_bytes, qber, protocol)
        self.keys: Dict[int, Tuple[bytes, float, str]] = {}
        # metriche time-series
        self.metrics: Dict[int, Dict[str, list]] = {}  # {neighbor: {"qber":[...], "noise":[...]}}

    def establish_key(self, neighbor: int, protocol: str = "bb84",
                      n_bits: int = 256, noise: float = 0.02, eve_prob: float = 0.0):
        if protocol == "bb84":
            key, qber = bb84_key_generation(n_bits=n_bits, noise_level=noise, eve=(random.random() < eve_prob))
        elif protocol == "e91":
            key, qber = e91_key_generation(n_bits=n_bits)
        else:
            raise ValueError("Unknown protocol")

        self.keys[neighbor] = (key, qber, protocol)
        self.metrics.setdefault(neighbor, {"qber": [], "noise": []})
        self.metrics[neighbor]["qber"].append(qber)
        self.metrics[neighbor]["noise"].append(noise)
        return key, qber

def build_network(num_nodes: int = 4,
                  noise_range=(0.0, 0.05),
                  eve_prob: float = 0.2):
    G = nx.complete_graph(num_nodes)
    nodes = {i: QKDNode(f"Node{i}") for i in G.nodes}
    for u, v in G.edges:
        noise = random.uniform(*noise_range)
        protocol = random.choice(["bb84", "e91"])
        key, qber = nodes[u].establish_key(v, protocol=protocol, noise=noise, eve_prob=eve_prob)
        nodes[v].keys[u] = (key, qber, protocol)
        nodes[v].metrics[u] = nodes[u].metrics[v]
        print(f"[Link {u}-{v}] Proto={protocol}, QBER={qber:.2%}, noise≈{noise:.3f}")
    return G, nodes
