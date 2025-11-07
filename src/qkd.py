# qkd.py (fast BB84)
from __future__ import annotations
import random
from hashlib import sha256
from typing import Tuple, List

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# --- create simulator & noise once ---
_NOISE = NoiseModel()
# we’ll attach gate noise names once at import; tweak level at call time via parameter on gates
# (Aer doesn’t let you change error probs on-the-fly; simplest is rebuild NoiseModel if you vary it a lot)
def _make_noise_model(p: float) -> NoiseModel:
    nm = NoiseModel()
    if p > 0:
        err = depolarizing_error(p, 1)
        for g in ["x", "h", "id"]:
            nm.add_all_qubit_quantum_error(err, g)
    return nm

# keep one simulator; we’ll swap noise by constructing a new simulator when p changes
_SIM_CACHE: dict[float, AerSimulator] = {}
def _sim_for_noise(p: float) -> AerSimulator:
    if p not in _SIM_CACHE:
        _SIM_CACHE[p] = AerSimulator(noise_model=_make_noise_model(p))
        # you can also try: _SIM_CACHE[p].set_options(max_parallel_threads=0)  # use all cores
    return _SIM_CACHE[p]

def _bits_to_bytes(bits: List[int]) -> bytes:
    pad = (-len(bits)) % 8
    bits_padded = bits + [0]*pad
    out = bytearray()
    for i in range(0, len(bits_padded), 8):
        byte = 0
        for b in bits_padded[i:i+8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)

def bb84_key_generation(n_bits: int = 256,
                        noise_level: float = 0.0,
                        eve: bool = False,
                        qber_threshold: float = 0.11,
                        raise_on_high_qber: bool = False) -> Tuple[bytes | None, float]:
    """
    Fast BB84 with chunking to avoid CircuitTooWideForTarget.
    Builds multiple n_chunk-qubit circuits if needed, runs each once,
    then concatenates results.
    """
    rng = random
    alice_bits  = np.fromiter((rng.randint(0, 1) for _ in range(n_bits)), dtype=np.int8)
    alice_bases = np.fromiter((rng.randint(0, 1) for _ in range(n_bits)), dtype=np.int8)  # 0=Z, 1=X
    bob_bases   = np.fromiter((rng.randint(0, 1) for _ in range(n_bits)), dtype=np.int8)

    backend = _sim_for_noise(noise_level)

    # --- detect max qubits available on this backend ---
    try:
        max_qubits = int(getattr(getattr(backend, "target", None), "num_qubits", None))  # qiskit >=1.0
        if not max_qubits or max_qubits <= 0:
            raise AttributeError
    except Exception:
        max_qubits = 28  # safe default under the reported 29 cap

    # --- helper to run one chunk and return Bob's bits as np.array[0/1] ---
    def run_chunk(a_bits: np.ndarray, a_bases: np.ndarray, b_bases: np.ndarray) -> np.ndarray:
        m = a_bits.shape[0]
        qc = QuantumCircuit(m, m)

        # Alice preparation
        for i in np.where(a_bits == 1)[0]:
            qc.x(i)
        for i in np.where(a_bases == 1)[0]:
            qc.h(i)

        # Eve disturbance (approximate, fast)
        if eve:
            eve_bases = np.fromiter((rng.randint(0, 1) for _ in range(m)), dtype=np.int8)
            # introduce flips with p=0.5 where bases differ (models intercept-resend disturbance)
            for i in np.where(eve_bases != a_bases)[0]:
                if rng.random() < 0.5:
                    qc.x(i)

        # Bob rotates and measures
        for i in np.where(b_bases == 1)[0]:
            qc.h(i)
        qc.measure(range(m), range(m))

        tqc = transpile(qc, backend)
        res = backend.run(tqc, shots=1, memory=True).result()
        bits = res.get_memory()[0][::-1]  # reverse to align index→qubit
        return (np.frombuffer(bits.encode(), dtype="S1").astype(np.uint8) - ord("0"))

    # --- run in chunks and concatenate ---
    bob_results_list: list[np.ndarray] = []
    start = 0
    while start < n_bits:
        end = min(start + max_qubits, n_bits)
        bob_results_list.append(
            run_chunk(alice_bits[start:end], alice_bases[start:end], bob_bases[start:end])
        )
        start = end
    bob_results = np.concatenate(bob_results_list, axis=0)

    # --- sifting & QBER ---
    mask = (alice_bases == bob_bases)
    if not mask.any():
        return None, 1.0

    sift_idx = np.where(mask)[0]
    sample_size = max(3, len(sift_idx) // 10)
    sample_idx = rng.sample(list(sift_idx), k=min(sample_size, len(sift_idx)))
    errors = int(np.sum(alice_bits[sample_idx] != bob_results[sample_idx]))
    qber = errors / len(sample_idx)

    if qber > qber_threshold:
        if raise_on_high_qber:
            raise ValueError(f"Eavesdropper/rumore eccessivo: QBER={qber:.2%}")
        return None, qber

    # --- derive key (simple privacy amplification) ---
    agree = (alice_bits[sift_idx] == bob_results[sift_idx])
    key_bits = alice_bits[sift_idx][agree].tolist()
    if len(key_bits) < 8:
        return None, qber
    raw = _bits_to_bytes(key_bits[:256])
    key = sha256(raw).digest()[:16]
    return key, qber



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
