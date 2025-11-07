# workbench.py
from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def _rng(seed: Optional[int]):
    if seed is None:
        return random.Random()
    r = random.Random()
    r.seed(seed)
    return r

def run_bb84_trace(key_len: int, noise_level: float, eve_prob: float,
                   sample_frac: float, seed: Optional[int] = None) -> Dict:
    R = _rng(seed)
    alice_bits = [R.randint(0,1) for _ in range(key_len)]
    alice_bases = [R.choice(["Z","X"]) for _ in range(key_len)]
    bob_bases   = [R.choice(["Z","X"]) for _ in range(key_len)]

    # Prepare qubits
    qcs = []
    for a_bit, a_basis in zip(alice_bits, alice_bases):
        qc = QuantumCircuit(1,1)
        if a_bit == 1:
            qc.x(0)
        if a_basis == "X":
            qc.h(0)
        qcs.append(qc)

    # Eve (intercept-resend) with probability eve_prob per qubit
    simulator = AerSimulator()
    eve_indices = [i for i in range(key_len) if R.random() < eve_prob]
    eve_bases = {}
    if eve_indices:
        circ_eve = []
        for i in eve_indices:
            tmp = qcs[i].copy()
            eve_bases[i] = R.choice(["Z","X"])
            if eve_bases[i] == "X":
                tmp.h(0)
            tmp.measure(0,0)
            circ_eve.append(tmp)
        tr = transpile(circ_eve, simulator)
        res = simulator.run(tr, shots=1, memory=True).result()
        # reconstruct & replace
        for k, i in enumerate(eve_indices):
            meas = int(res.get_memory(k)[0])
            rebuilt = QuantumCircuit(1,1)
            if meas == 1:
                rebuilt.x(0)
            if eve_bases[i] == "X":
                rebuilt.h(0)
            qcs[i] = rebuilt

    # Bob measure
    circ_bob = []
    for i in range(key_len):
        qc = qcs[i]
        if bob_bases[i] == "X":
            qc.h(0)
        qc.measure(0,0)
        circ_bob.append(qc)
    trb = transpile(circ_bob, simulator)
    resb = simulator.run(trb, shots=1, memory=True).result()
    bob_results = [int(resb.get_memory(i)[0]) for i in range(key_len)]

    # Sifting
    sift_mask = [i for i in range(key_len) if alice_bases[i] == bob_bases[i]]
    sifted_alice = [alice_bits[i] for i in sift_mask]
    sifted_bob   = [bob_results[i] for i in sift_mask]

    # QBER sample
    n_sample = max(1, int(len(sift_mask)*sample_frac))
    sample_idx = sorted(random.sample(range(len(sift_mask)), min(n_sample, len(sift_mask))))
    errors = sum(1 for j in sample_idx if sifted_alice[j] != sifted_bob[j])
    qber = errors / len(sample_idx) if sample_idx else 1.0

    # Remove sampled indices from final key
    keep_mask = [i for i in range(len(sift_mask)) if i not in sample_idx]
    final_alice = [sifted_alice[i] for i in keep_mask]
    final_bob   = [sifted_bob[i]   for i in keep_mask]

    return {
        "protocol": "bb84",
        "params": {
            "key_len": key_len,
            "noise_level": noise_level,
            "eve_prob": eve_prob,
            "sample_frac": sample_frac,
            "seed": seed
        },
        "trace": {
            "alice_bits": alice_bits,
            "alice_bases": alice_bases,
            "bob_bases": bob_bases,
            "eve_indices": eve_indices,
            "eve_bases": eve_bases,  # {idx: 'Z'|'X'}
            "bob_results": bob_results,
            "sift_indices": sift_mask,
            "sample_indices": sample_idx,
            "qber": qber,
            "final_key_alice": final_alice,
            "final_key_bob": final_bob,
            "keys_match": final_alice == final_bob
        }
    }

def run_e91_trace(key_len: int, seed: Optional[int] = None) -> Dict:
    # Stub ragionato (per eventuale estensione reale con CHSH)
    R = _rng(seed)
    pseudo = [R.randint(0,1) for _ in range(key_len)]
    return {
        "protocol": "e91",
        "params": {"key_len": key_len, "seed": seed},
        "trace": {
            "entangled_pairs": key_len,
            "settings_a": [R.choice([0,1,2]) for _ in range(key_len)],
            "settings_b": [R.choice([0,1,2]) for _ in range(key_len)],
            "raw_correlations": pseudo,
            "sift_indices": list(range(0, key_len, 2)),
            "qber": 0.0,
            "final_key_alice": pseudo[::2],
            "final_key_bob":  pseudo[::2],
            "keys_match": True
        }
    }
