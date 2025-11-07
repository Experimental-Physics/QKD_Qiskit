# qkd.py
from __future__ import annotations
import random
from hashlib import sha256
from typing import Tuple, List

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ------------------------ QRNG ------------------------

def qrng(n: int) -> str:
    """
    Quantum RNG using Qiskit backend.run() (execute() deprecated).
    Returns a bitstring of length n.
    """
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))

    backend = Aer.get_backend("qasm_simulator")  # modern Qiskit Aer backend
    job = backend.run(qc, shots=1)
    result = job.result()

    bits = list(result.get_counts().keys())[0]
    return bits[::-1]  # reverse ordering

# ------------------------ Utility ------------------------

def bits_to_bytes(bits: List[int]) -> bytes:
    """Converte una lista di bit [0/1] in bytes (padding a multipli di 8)."""
    pad = (-len(bits)) % 8
    bits_padded = bits + [0]*pad
    out = bytearray()
    for i in range(0, len(bits_padded), 8):
        byte = 0
        for b in bits_padded[i:i+8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)

# ------------------------ BB84 ------------------------

def bb84_key_generation(n_bits: int = 256,
                        noise_level: float = 0.0,
                        eve: bool = False,
                        qber_threshold: float = 0.11,
                        raise_on_high_qber: bool = False) -> Tuple[bytes | None, float]:
    """
    Simulazione BB84 base con rumore depolarizzante a 1 qubit.
    Ritorna (chiave AES-128 derivata o None se QBER alto, QBER stimato).

    Se raise_on_high_qber=True, solleva ValueError quando QBER supera qber_threshold.
    """
    # 1) Alice: bit e basi
    alice_bits = [random.randint(0, 1) for _ in range(n_bits)]
    alice_bases = [random.randint(0, 1) for _ in range(n_bits)]

    # 2) Preparazione stati (come prima)
    circuits = []
    for bit, basis in zip(alice_bits, alice_bases):
        qc = QuantumCircuit(1, 1)
        if bit == 1:
            qc.x(0)
        if basis == 1:
            qc.h(0)

        if eve:
            eve_basis = random.randint(0, 1)
            if eve_basis == 1:
                qc.h(0)
            qc.measure(0, 0)
            qc.reset(0)
            measured_bit = random.randint(0, 1) if eve_basis != basis else bit
            if measured_bit == 1:
                qc.x(0)
            if eve_basis == 1:
                qc.h(0)
        circuits.append(qc)

    # 3) Rumore depolarizzante
    noise = NoiseModel()
    err1 = depolarizing_error(noise_level, 1)
    for gate in ["x", "h", "id"]:
        noise.add_all_qubit_quantum_error(err1, gate)
    backend = AerSimulator(noise_model=noise)

    # 4) Bob
    bob_bases = [random.randint(0, 1) for _ in range(n_bits)]
    bob_results = []
    for i, qc in enumerate(circuits):
        qcm = qc.copy()
        if bob_bases[i] == 1:
            qcm.h(0)
        qcm.measure(0, 0)
        result = backend.run(transpile(qcm, backend), shots=1).result()
        bob_results.append(int(list(result.get_counts().keys())[0]))

    # 5) Sifting
    sifted_idx = [i for i in range(n_bits) if alice_bases[i] == bob_bases[i]]
    if not sifted_idx:
        return None, 1.0

    # 6) QBER su campione (più stabile)
    sample_size = max(3, len(sifted_idx) // 10)
    sample_idx = random.sample(sifted_idx, k=min(sample_size, len(sifted_idx)))
    errors = sum(int(alice_bits[j] != bob_results[j]) for j in sample_idx)
    qber = errors / len(sample_idx)

    # 7) Gestione QBER alto
    if qber > qber_threshold:
        if raise_on_high_qber:
            raise ValueError(f"Eavesdropper/rumore eccessivo: QBER={qber:.2%}")
        return None, qber

    # 8) Privacy amplification minimale
    sifted_alice = [alice_bits[i] for i in sifted_idx]
    sifted_bob   = [bob_results[i] for i in sifted_idx]
    key_bits = [a for a, b in zip(sifted_alice, sifted_bob) if a == b]
    if len(key_bits) < 8:
        # Non abbastanza materiale
        return None, qber
    raw_bytes = bits_to_bytes(key_bits[:256])
    final_key = sha256(raw_bytes).digest()[:16]
    return final_key, qber


# ------------------------ E91 (stub ragionato) ------------------------

def e91_key_generation(n_bits: int = 256) -> Tuple[bytes, float]:
    """
    Placeholder minimal per confronto. In un’implementazione completa: si preparano
    coppie entangled e si misurano in basi diverse con test CHSH.
    Qui ritorniamo chiave fittizia e QBER ~0 per demo.
    """
    # In una demo reale potresti chiamare AerSimulator con 2 qubit entangled e misure.
    fake_material = b"\x42" * 64
    return sha256(fake_material).digest()[:16], 0.0
