#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import time
from random import choice, randint, random, sample

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# --- NUOVO IMPORT PER IL RUMORE ---
from qiskit_aer.noise import NoiseModel, depolarizing_error

# (Il resto delle funzioni 'alice', 'estimate_qber' rimane invariato)
# ... (Puoi copiare le tue funzioni 'alice' e 'estimate_qber' qui) ...


def alice(q_channel, c_channel, key_len=10):
    """
    Alice's side of the protocol.
    (Questa funzione è invariata rispetto alla tua versione)
    """
    print(f"[Alice] Starting. Generating {key_len} bits and bases.")

    # 1. Generate bits and bases
    alice_bits = [randint(0, 1) for _ in range(key_len)]
    alice_bases = [choice(["Z", "X"]) for _ in range(key_len)]

    # 2. Create and "send" qubits
    q_channel.clear()
    for i in range(key_len):
        qc = QuantumCircuit(1, 1)
        if alice_bits[i] == 1:
            qc.x(0)
        if alice_bases[i] == "X":
            qc.h(0)
        q_channel.append(qc)

    print(f"[Alice] Sent {len(q_channel)} qubits on the quantum channel.")

    # 3. Wait for Bob's bases
    print("[Alice] Waiting for Bob to publish his bases on the classical channel...")
    while "bob_bases" not in c_channel:
        time.sleep(0.1)

    bob_bases = c_channel["bob_bases"]
    print("[Alice] Received Bob's bases.")

    # 4. Publish her bases
    c_channel["alice_bases"] = alice_bases

    # 5. Sift her key
    sifted_alice_key = []
    for i in range(key_len):
        if alice_bases[i] == bob_bases[i]:
            sifted_alice_key.append(alice_bits[i])

    print(f"[Alice] Sifted key: {''.join(map(str, sifted_alice_key))}")
    return alice_bits, sifted_alice_key


# --- NUOVA FUNZIONE PER CREARE IL MODELLO DI RUMORE QISKIT ---
def create_hardware_noise_model(prob_depolarizing):
    """
    Crea un semplice modello di rumore hardware (rumore depolarizzante)
    da applicare ai gate single-qubit.
    """
    if prob_depolarizing == 0:
        return None  # Nessun rumore

    # L'errore depolarizzante simula una perdita generica di coerenza
    error_1 = depolarizing_error(prob_depolarizing, 1)

    # Crea un modello di rumore vuoto
    noise_model = NoiseModel()

    # Applica questo errore a tutti i gate single-qubit che Alice usa
    noise_model.add_quantum_error(error_1, ["x", "h"], [0])

    print(
        f"[Noise] Created hardware noise model with {prob_depolarizing * 100:.1f}% depolarizing error on X and H gates."
    )
    return noise_model


# --- FUNZIONE 'bob' MODIFICATA ---
def bob(q_channel, c_channel, key_len=10, eve_intercept_prob=0.0, noise_model=None):
    """
    Bob's side of the protocol.
    Ora accetta:
    - eve_intercept_prob: La probabilità che Eve esegua un attacco (ex 'noise_level')
    - noise_model: Il modello di rumore hardware di Qiskit
    """
    print(f"[Bob] Starting. (Eve Intercept Prob: {eve_intercept_prob * 100}%)")

    # --- MODIFICA: Il simulatore ora accetta il modello di rumore ---
    simulator = AerSimulator(noise_model=noise_model)

    # 1. Simulate Eve/Noise (attacco 'intercept-and-resend')
    # (Questa logica è ottimizzata e usa 'eve_intercept_prob')
    if eve_intercept_prob > 0:
        print("[Bob/Eve] Eavesdropper is intercepting the quantum channel...")
        intercept_info = []
        circuits_for_eve = []

        for i in range(key_len):
            if random() < eve_intercept_prob:  # Eve intercetta
                eve_basis = choice(["Z", "X"])
                intercept_info.append({"index": i, "basis": eve_basis})
                qc_from_alice = q_channel[i].copy()
                if eve_basis == "X":
                    qc_from_alice.h(0)
                qc_from_alice.measure(0, 0)
                circuits_for_eve.append(qc_from_alice)

        if circuits_for_eve:
            print(
                f"[Bob/Eve] Eve is measuring {len(circuits_for_eve)} qubits in one batch..."
            )
            # IMPORTANTE: La simulazione di Eve è ORA influenzata dal rumore hardware!
            transpiled_eve = transpile(circuits_for_eve, simulator)
            result_eve = simulator.run(transpiled_eve, shots=1, memory=True).result()
            measured_bits = [
                int(result_eve.get_memory(k)[0]) for k in range(len(circuits_for_eve))
            ]

            for k, info in enumerate(intercept_info):
                original_index = info["index"]
                eve_basis = info["basis"]
                eve_measured_bit = measured_bits[k]
                qc_for_bob = QuantumCircuit(1, 1)
                if eve_measured_bit == 1:
                    qc_for_bob.x(0)
                if eve_basis == "X":
                    qc_for_bob.h(0)
                q_channel[original_index] = qc_for_bob

    # 2. Generate Bob's bases
    print(f"[Bob] Receiving {len(q_channel)} qubits...")
    bob_bases = [choice(["Z", "X"]) for _ in range(key_len)]

    # 3. Measure qubits
    circuits_to_run = []
    for i in range(key_len):
        qc = q_channel[i]
        if bob_bases[i] == "X":
            qc.h(0)
        qc.measure(0, 0)
        circuits_to_run.append(qc)

    # IMPORTANTE: Anche la simulazione di Bob è influenzata dallo stesso rumore hardware
    transpiled_circuits = transpile(circuits_to_run, simulator)
    result = simulator.run(transpiled_circuits, shots=1, memory=True).result()

    bob_results = []
    for i in range(len(circuits_to_run)):
        bob_results.append(int(result.get_memory(i)[0]))

    print(f"[Bob] Measured results: {''.join(map(str, bob_results))}")

    # 4. Publish his bases
    c_channel["bob_bases"] = bob_bases
    print("[Bob] Published his bases on the classical channel.")

    # 5. Wait for Alice's bases
    print("[Bob] Waiting for Alice to publish her bases...")
    while "alice_bases" not in c_channel:
        time.sleep(0.1)

    alice_bases = c_channel["alice_bases"]
    print("[Bob] Received Alice's bases.")

    # 6. Sift his key
    sifted_bob_key = []
    for i in range(key_len):
        if bob_bases[i] == alice_bases[i]:
            sifted_bob_key.append(bob_results[i])

    print(f"[Bob] Sifted key: {''.join(map(str, sifted_bob_key))}")
    return bob_results, sifted_bob_key


def estimate_qber(sifted_alice, sifted_bob, sample_size=0.5):
    """
    (Questa funzione è invariata rispetto alla tua versione)
    """
    if not sifted_alice or not sifted_bob:
        return 0.0, [], []

    num_samples = int(len(sifted_alice) * sample_size)
    if num_samples == 0:
        print("[QBER] Sifted key is too short to sample. Aborting.")
        return 0.0, [], []

    sample_indices = sample(range(len(sifted_alice)), num_samples)
    sample_indices.sort(reverse=True)

    print(f"[QBER] Comparing {num_samples} bits to estimate error rate...")
    errors = 0
    final_alice = list(sifted_alice)
    final_bob = list(sifted_bob)

    for i in sample_indices:
        alice_sample_bit = final_alice.pop(i)
        bob_sample_bit = final_bob.pop(i)
        if alice_sample_bit != bob_sample_bit:
            errors += 1

    if num_samples == 0:
        return 0.0, final_alice, final_bob

    qber = errors / num_samples
    return qber, final_alice, final_bob


# --- FUNZIONE 'run_simulation' MODIFICATA ---
def run_simulation(key_len, eve_intercept_prob, noise_model):
    """
    Runs a single simulation.
    Ora passa 'eve_intercept_prob' e 'noise_model' a Bob.
    """
    q_channel = []
    c_channel = {}
    results = {}

    alice_thread = threading.Thread(
        target=lambda: results.update({"alice": alice(q_channel, c_channel, key_len)})
    )
    # --- MODIFICA: Passa i nuovi parametri a bob ---
    bob_thread = threading.Thread(
        target=lambda: results.update(
            {"bob": bob(q_channel, c_channel, key_len, eve_intercept_prob, noise_model)}
        )
    )

    alice_thread.start()
    time.sleep(0.1)
    bob_thread.start()
    alice_thread.join()
    bob_thread.join()

    if "alice" not in results or "bob" not in results:
        print("[Error] A thread failed to return results. Aborting run.")
        return 0.0, 0

    _, sifted_alice_key = results["alice"]
    _, sifted_bob_key = results["bob"]

    print("\n--- Sifting Complete ---")
    print(
        f"Alice Sifted Key ({len(sifted_alice_key)}): {''.join(map(str, sifted_alice_key))}"
    )
    print(
        f"Bob Sifted Key   ({len(sifted_bob_key)}): {''.join(map(str, sifted_bob_key))}"
    )

    qber, final_alice, final_bob = estimate_qber(sifted_alice_key, sifted_bob_key)

    print("\n--- Final Analysis ---")
    print(f"Initial Key Length:    {key_len}")
    print(f"Sifted Key Length:     {len(sifted_alice_key)}")
    print(f"Final Key Length:      {len(final_alice)}")
    print(f"Quantum Bit Error Rate (QBER): {qber:.2%}")

    if qber > 0.15:
        print("!! QBER IS TOO HIGH! EAVESDROPPING DETECTED. KEY DISCARDED. !!")
        return qber, 0
    else:
        print(">> QBER is low. Secure key established.")
        print(f">> Final Shared Key:  {''.join(map(str, final_alice))}")
        print(f">> Keys Match:        {final_alice == final_bob}")
        return qber, len(final_alice)


# --- FUNZIONE 'main' MODIFICATA PER GESTIRE I NUOVI PARAMETRI ---
def main():
    # Aumentiamo KEY_LENGTH per vedere meglio l'effetto statistico del rumore
    # NOTA: questo renderà l'esecuzione più lenta! Riduci 'range(3)' sotto se troppo lento.
    KEY_LENGTH = 100  # Number of qubits to send

    # --- Definiamo i nostri due nuovi parametri di rumore ---
    EVE_PROB_SCENARIO_2 = 1.0  # Per lo Scenario 2 (attacco totale)
    HARDWARE_NOISE_NONE = 0.0  # 0% rumore hardware
    HARDWARE_NOISE_REALISTIC = 0.03  # 3% rumore hardware

    # Creiamo i modelli di rumore UNA SOLA VOLTA
    noise_model_none = create_hardware_noise_model(HARDWARE_NOISE_NONE)  # Ritorna None
    noise_model_realistic = create_hardware_noise_model(HARDWARE_NOISE_REALISTIC)

    # --- Scenario 1: Canale sicuro E hardware ideale ---
    print("=================================================")
    print(f" SCENARIO 1: Secure Channel (Eve=0%) + Ideal Hardware (Noise=0%) ")
    print("=================================================")
    run_simulation(
        key_len=KEY_LENGTH, eve_intercept_prob=0.0, noise_model=noise_model_none
    )

    # --- Scenario 2: Attacco totale MA hardware ideale ---
    print("\n=================================================")
    print(f" SCENARIO 2: Full Eavesdropping (Eve=100%) + Ideal Hardware (Noise=0%) ")
    print("=================================================")
    run_simulation(
        key_len=KEY_LENGTH,
        eve_intercept_prob=EVE_PROB_SCENARIO_2,
        noise_model=noise_model_none,
    )

    # --- Scenario 3: Canale sicuro MA hardware rumoroso ---
    print("\n=================================================")
    print(
        f" SCENARIO 3: Secure Channel (Eve=0%) + Realistic Hardware (Noise={HARDWARE_NOISE_REALISTIC * 100}%) "
    )
    print("=================================================")
    run_simulation(
        key_len=KEY_LENGTH, eve_intercept_prob=0.0, noise_model=noise_model_realistic
    )

    # --- Scenario 4: Plotting QBER vs. Eve (confrontando hardware ideale e rumoroso) ---
    print("\n=================================================")
    print(" SCENARIO 4: Plotting QBER vs. Eve Interception ")
    print("=================================================")
    print("Running multiple simulations... (This may take a moment)")

    # L'asse X è la probabilità di intercettazione di Eve
    eve_prob_levels = np.linspace(0, 1, 15)  # 15 punti per il grafico
    avg_qbers_ideal_hw = []
    avg_qbers_noisy_hw = []

    # Eseguiamo la simulazione per entrambi i modelli di hardware
    for eve_prob in eve_prob_levels:
        print(f"\nSimulating Eve Intercept Prob: {eve_prob:.2f}")

        # 1. Esegui con hardware IDEALE
        print(f"... simulating ideal hardware (0% noise)...")
        qbers_ideal = []
        for _ in range(3):  # Media su 3 run per velocità
            qber, _ = run_simulation(
                key_len=KEY_LENGTH,
                eve_intercept_prob=eve_prob,
                noise_model=noise_model_none,
            )
            qbers_ideal.append(qber)
        avg_qbers_ideal_hw.append(np.mean(qbers_ideal))

        # 2. Esegui con hardware RUMOROSO
        print(
            f"... simulating noisy hardware ({HARDWARE_NOISE_REALISTIC * 100}% noise)..."
        )
        qbers_noisy = []
        for _ in range(3):  # Media su 3 run per velocità
            qber, _ = run_simulation(
                key_len=KEY_LENGTH,
                eve_intercept_prob=eve_prob,
                noise_model=noise_model_realistic,
            )
            qbers_noisy.append(qber)
        avg_qbers_noisy_hw.append(np.mean(qbers_noisy))

    # Plotting
    plt.figure(figsize=(10, 7))

    # Linea 1: Hardware Ideale (dovrebbe partire da QBER=0)
    plt.plot(
        eve_prob_levels,
        avg_qbers_ideal_hw,
        "bo-",
        label="Simulated QBER (Ideal Hardware, 0% Noise)",
    )

    # Linea 2: Hardware Rumoroso (dovrebbe partire da QBER > 0)
    plt.plot(
        eve_prob_levels,
        avg_qbers_noisy_hw,
        "go-",
        label=f"Simulated QBER ({HARDWARE_NOISE_REALISTIC * 100}% Hardware Noise)",
    )

    # Linea 3: Limite teorico (solo Eve)
    plt.plot(
        eve_prob_levels,
        eve_prob_levels * 0.25,
        "r--",
        label="Theoretical QBER (Eve-only, 0.25 * P(Eve))",
    )

    plt.title("BB84: QBER vs. Eve Interception (with Hardware Noise)")
    plt.xlabel("Eve Interception Probability (eve_intercept_prob)")
    plt.ylabel("Quantum Bit Error Rate (QBER)")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)  # Forza l'asse Y a partire da 0

    filename = "bb84_qber_vs_noise_comparison.png"
    plt.savefig(filename)
    print(f"\n[Plotting] Grafico di confronto salvato come: {filename}")


if __name__ == "__main__":
    main()
