
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
    backend = Aer.get_backend('qasm_simulator')

    # Bob misura con basi random
    bob_bases = [random.randint(0, 1) for _ in range(n_bits)]
    bob_results = []
    for i, qc in enumerate(circuits):
        if bob_bases[i] == 1:
            qc.h(0)
        qc.measure(0, 0)
        result = execute(qc, backend, noise_model=noise_model, shots=1).result()
        bob_results.append(int(list(result.get_counts().keys())[0]))

    # Sifting: confronta basi (canale classico simulato)
    sifted_key = []
    for i in range(n_bits):
        if alice_bases[i] == bob_bases[i]:
            sifted_key.append(alice_bits[i])

    # Error estimation: sample 10% per QBER
    sample_size = max(1, len(sifted_key) // 10)
    errors = sum(alice_bits[i] != bob_results[i] for i in random.sample(range(n_bits), sample_size) if alice_bases[i] == bob_bases[i])
    qber = errors / sample_size if sample_size > 0 else 0

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