from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import random
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

def bb84_key_generation(n_bits=256, noise_level=0.0, eve=False):
    # Alice genera bit e basi random
    alice_bits = [random.randint(0, 1) for _ in range(n_bits)]
    alice_bases = [random.randint(0, 1) for _ in range(n_bits)]  # 0: rectilinear, 1: diagonal
    
    # Prepara qubit
    circuits = []
    for bit, basis in zip(alice_bits, alice_bases):
        qc = QuantumCircuit(1, 1)
        if bit == 1:
            qc.x(0)
        if basis == 1:
            qc.h(0)
        if eve:  # Eve intercetta
            eve_basis = random.randint(0, 1)
            if eve_basis == 1:
                qc.h(0)
            qc.measure(0, 0)
            # Simula re-preparazione (semplificato)
            qc.reset(0)
            if random.randint(0, 1) == 1:
                qc.x(0)
            if eve_basis == 1:
                qc.h(0)
        circuits.append(qc)
    
    # Aggiungi noise
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(noise_level, 1), ['u1', 'u2', 'u3'])
    
    # Crea il backend con noise
    backend = AerSimulator(noise_model=noise_model)
    
    # Bob misura con basi random
    bob_bases = [random.randint(0, 1) for _ in range(n_bits)]
    bob_results = []
    for i, qc in enumerate(circuits):
        if bob_bases[i] == 1:
            qc.h(0)
        qc.measure(0, 0)
        # Transpile per il backend (opzionale ma raccomandato)
        qc_transpiled = transpile(qc, backend)
        result = backend.run(qc_transpiled, shots=1).result()
        bob_results.append(int(list(result.get_counts().keys())[0]))
    
    # Sifting: confronta basi (canale classico simulato)
    sifted_key = []
    sifted_indices = []
    for i in range(n_bits):
        if alice_bases[i] == bob_bases[i]:
            sifted_key.append(alice_bits[i])
            sifted_indices.append(i)
    
    # Error estimation: sample 10% per QBER
    sample_size = max(1, len(sifted_key) // 10)
    if len(sifted_indices) > 0:
        sample = random.sample(sifted_indices, min(sample_size, len(sifted_indices)))
        errors = sum(alice_bits[j] != bob_results[j] for j in sample)
        qber = errors / len(sample) if len(sample) > 0 else 0
    else:
        qber = 0
    
    if qber > 0.11:  # Threshold per rilevare Eve
        raise ValueError("Eavesdropper detected! QBER too high: {:.2%}".format(qber))
    
    # Privacy amplification (semplificato: hash la key)
    from hashlib import sha256
    key_bytes = bytes(sifted_key[:128])  # Truncia a 128 bit per AES-128
    final_key = sha256(key_bytes).digest()[:16]  # 128-bit key
    return final_key, qber

# Esempio uso
key, qber = bb84_key_generation(512, noise_level=0.05, eve=False)
print("Chiave generata:", key.hex(), "QBER:", qber)