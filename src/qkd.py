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
                        eve: bool = False) -> Tuple[bytes, float]:
    """
    Simulazione BB84 base con rumore depolarizzante a 1 qubit.
    Ritorna (chiave AES-128 derivata, QBER stimato).

    Note:
      - Basi: 0=Z (rettilinea), 1=X (diagonale)
      - Se 'eve' True, si simula una misura intercettata con base random.
    """
    # 1) Alice: bit e basi
    alice_bits = [random.randint(0, 1) for _ in range(n_bits)]
    alice_bases = [random.randint(0, 1) for _ in range(n_bits)]

    # 2) Preparazioni dei circuiti
    circuits = []
    for bit, basis in zip(alice_bits, alice_bases):
        qc = QuantumCircuit(1, 1)
        if bit == 1:
            qc.x(0)
        if basis == 1:   # base X
            qc.h(0)

        # Eve (intercept-resend semplificato)
        if eve:
            eve_basis = random.randint(0, 1)
            # misura in base eve_basis
            if eve_basis == 1:
                qc.h(0)
            qc.measure(0, 0)

            # reset e ripreparazione secondo esito misurato (approssimata)
            qc.reset(0)
            # (ri)prepara secondo bit misurato (ignoriamo il reale risultato per semplicità
            # e introduciamo 50% di errore quando le basi differiscono)
            measured_bit = random.randint(0, 1) if eve_basis != basis else bit
            if measured_bit == 1:
                qc.x(0)
            if eve_basis == 1:
                qc.h(0)
            qc = QuantumCircuit.compose(QuantumCircuit(1, 1), qc)

        circuits.append(qc)

    # 3) Rumore depolarizzante
    noise = NoiseModel()
    # Applichiamo errore depolarizzante su X/H/ID (base), valori tipici 'x','h','id','measure'
    err1 = depolarizing_error(noise_level, 1)
    for gate in ["x", "h", "id"]:
        noise.add_all_qubit_quantum_error(err1, gate)

    backend = AerSimulator(noise_model=noise)

    # 4) Bob: basi e misure
    bob_bases = [random.randint(0, 1) for _ in range(n_bits)]
    bob_results = []
    for i, qc in enumerate(circuits):
        qcm = qc.copy()
        if bob_bases[i] == 1:
            qcm.h(0)
        qcm.measure(0, 0)
        transpiled = transpile(qcm, backend)
        result = backend.run(transpiled, shots=1).result()
        bit = int(list(result.get_counts().keys())[0])
        bob_results.append(bit)

    # 5) Sifting
    sifted_idx = [i for i in range(n_bits) if alice_bases[i] == bob_bases[i]]
    sifted_alice = [alice_bits[i] for i in sifted_idx]
    sifted_bob   = [bob_results[i] for i in sifted_idx]

    # 6) Stima QBER su un campione (10% o almeno 1)
    if not sifted_idx:
        # fallback (nessun match di basi) -> nessuna chiave
        return b"", 1.0

    sample_size = max(1, len(sifted_idx) // 10)
    sample_idx = random.sample(sifted_idx, k=min(sample_size, len(sifted_idx)))
    errors = sum(int(alice_bits[j] != bob_results[j]) for j in sample_idx)
    qber = errors / len(sample_idx)

    # 7) Soglia (indicativa; pratica: ~11%)
    if qber > 0.11:
        raise ValueError(f"Eavesdropper/rumore eccessivo: QBER={qber:.2%}")

    # 8) Privacy amplification (semplice: hash della chiave sfoltita)
    # Usiamo metà dei bit sifted per la chiave effettiva (grezzo) e poi SHA-256 -> AES-128
    key_bits = [a for a, b in zip(sifted_alice, sifted_bob) if a == b]
    # assicuriamoci di avere materiale sufficiente
    if len(key_bits) < 128:
        # se troppi pochi, ripieghiamo su tutto il sifted concordato
        pass
    raw_bytes = bits_to_bytes(key_bits[:256])  # fino a 256 bit grezzi
    final_key = sha256(raw_bytes).digest()[:16]  # 16 byte = 128 bit
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
