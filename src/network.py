# network.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import networkx as nx

from qkd import bb84_key_generation, e91_key_generation

@dataclass
class NetworkState:
    graph: nx.Graph
    nodes: Dict[int, "QKDNode"]

class QKDNode:
    def __init__(self, name: str):
        self.name = name
        # neighbor_id -> (key_bytes, qber, protocol)
        self.keys: Dict[int, Tuple[bytes, float, str]] = {}
        # neighbor_id -> {"qber":[...], "noise":[...]}
        self.metrics: Dict[int, Dict[str, list]] = {}

    def establish_key(self, neighbor: int, protocol: str = "bb84",
                      n_bits: int = 256, noise: float = 0.02,
                      eve_prob: float = 0.0, qber_threshold: float = 0.11):
        key: Optional[bytes] = None
        qber = 1.0
        chosen_proto = protocol.lower()

        if chosen_proto == "bb84":
            attempts = [
                dict(n_bits=n_bits, noise=noise, eve=(random.random() < eve_prob)),
                dict(n_bits=max(512, n_bits), noise=noise * 0.5, eve=False),
                dict(n_bits=1024, noise=noise * 0.25, eve=False),
            ]
            for params in attempts:
                key, qber = bb84_key_generation(
                    n_bits=params["n_bits"],
                    noise_level=max(0.0, params["noise"]),
                    eve=params["eve"],
                    qber_threshold=qber_threshold,
                    raise_on_high_qber=False
                )
                if key is not None:
                    break
            if key is None:
                chosen_proto = "e91"
                key, qber = e91_key_generation(n_bits=max(256, n_bits))

        elif chosen_proto == "e91":
            key, qber = e91_key_generation(n_bits=max(256, n_bits))
        else:
            chosen_proto = "bb84"
            key, qber = bb84_key_generation(n_bits=n_bits, noise_level=noise, eve=False)

        if key is not None:
            self.keys[neighbor] = (key, qber, chosen_proto)
            self.metrics.setdefault(neighbor, {"qber": [], "noise": []})
            self.metrics[neighbor]["qber"].append(qber)
            self.metrics[neighbor]["noise"].append(noise)

        return key, qber, chosen_proto

def build_network(num_nodes: int = 4,
                  noise_range=(0.0, 0.05),
                  eve_prob: float = 0.2,
                  default_protocol: str = "mixed") -> NetworkState:
    G = nx.complete_graph(num_nodes)
    nodes = {i: QKDNode(f"Node{i}") for i in G.nodes}
    for u, v in G.edges:
        noise = random.uniform(*noise_range)
        if default_protocol == "mixed":
            protocol = random.choice(["bb84", "e91"])
        else:
            protocol = default_protocol
        key, qber, proto = nodes[u].establish_key(
            v, protocol=protocol, n_bits=256, noise=noise,
            eve_prob=eve_prob
        )
        nodes[v].keys[u] = (key, qber, proto)
        nodes[v].metrics[u] = nodes[u].metrics[v]
        print(f"[Link {u}-{v}] Proto={proto}, QBER={qber:.2%}, noise≈{noise:.3f}")
    return NetworkState(graph=G, nodes=nodes)
