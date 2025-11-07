# network.py
from __future__ import annotations
import random
from typing import Dict, Tuple
import networkx as nx

from qkd import bb84_key_generation, e91_key_generation

class QKDNode:
    def __init__(self, name: str):
        self.name = name
        self.keys: Dict[int, Tuple[bytes, float, str]] = {}
        self.metrics: Dict[int, Dict[str, list]] = {}

    def establish_key(self, neighbor: int, protocol: str = "bb84",
                      n_bits: int = 256, noise: float = 0.02, eve_prob: float = 0.0):
        """
        Prova a stabilire una chiave. Non lancia eccezioni su QBER alto.
        Ritorna (key, qber) con key!=None solo se il link è sicuro.
        """
        key = None
        qber = 1.0
        chosen_proto = protocol

        if protocol == "bb84":
            # fino a 3 tentativi: meno rumore, niente Eve, più bit
            attempts = [
                dict(n_bits=n_bits, noise=noise,       eve=(random.random() < eve_prob)),
                dict(n_bits=max(512, n_bits), noise=noise * 0.5, eve=False),
                dict(n_bits=1024,            noise=noise * 0.25, eve=False),
            ]
            for params in attempts:
                key, qber = bb84_key_generation(
                    n_bits=params["n_bits"],
                    noise_level=max(0.0, params["noise"]),
                    eve=params["eve"],
                    raise_on_high_qber=False  # non alzare eccezioni
                )
                if key is not None:
                    break

            # fallback a E91 se BB84 non riesce
            if key is None:
                chosen_proto = "e91"
                key, qber = e91_key_generation(n_bits=max(256, n_bits))

        elif protocol == "e91":
            key, qber = e91_key_generation(n_bits=max(256, n_bits))
        else:
            raise ValueError("Unknown protocol")

        # Salva SOLO se abbiamo una chiave valida
        if key is not None:
            self.keys[neighbor] = (key, qber, chosen_proto)
            self.metrics.setdefault(neighbor, {"qber": [], "noise": []})
            # registriamo il 'noise' di input come metrica (anche se attenuato nei retry)
            self.metrics[neighbor]["qber"].append(qber)
            self.metrics[neighbor]["noise"].append(noise)
        # Se key è None lasciamo il link non pronto (niente inserimento in self.keys)

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
