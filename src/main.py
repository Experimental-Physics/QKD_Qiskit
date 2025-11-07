#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import time
from random import choice, randint, random, sample

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# We will use simple list and dict as our "connections"
# q_channel: A list shared between Alice and Bob, holding QuantumCircuits (the "qubits")
# c_channel: A dict shared between Alice and Bob, holding public classical messages
#            (e.g., 'bob_bases', 'alice_bases')


def alice(q_channel, c_channel, key_len=10):
    """
    Alice's side of the protocol.
    1. Generates her bits and bases.
    2. Creates and "sends" qubits (QuantumCircuits) via the q_channel.
    3. Waits for Bob to publish his bases on the c_channel.
    4. Publishes her own bases on the c_channel.
    5. Sifts her key based on matching bases.
    6. Returns her original bits and her final sifted key.
    """
    print(f"[Alice] Starting. Generating {key_len} bits and bases.")

    # 1. Generate bits and bases
    #    'Z' basis = Rectilinear (|0>, |1>)
    #    'X' basis = Diagonal (|+>, |->)
    alice_bits = [randint(0, 1) for _ in range(key_len)]
    alice_bases = [choice(["Z", "X"]) for _ in range(key_len)]

    # 2. Create and "send" qubits
    q_channel.clear()  # Clear the channel for this new transmission
    for i in range(key_len):
        qc = QuantumCircuit(1, 1)

        # Encode bit: 1 -> apply X gate
        if alice_bits[i] == 1:
            qc.x(0)

        # Encode basis: 'X' -> apply H gate
        if alice_bases[i] == "X":
            qc.h(0)

        q_channel.append(qc)  # "Send" the prepared qubit

    print(f"[Alice] Sent {len(q_channel)} qubits on the quantum channel.")

    # 3. Wait for Bob's bases
    print("[Alice] Waiting for Bob to publish his bases on the classical channel...")
    while "bob_bases" not in c_channel:
        time.sleep(0.1)  # Wait for Bob to finish measuring

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

    # Return both for analysis
    return alice_bits, sifted_alice_key


def bob(q_channel, c_channel, key_len=10, noise_level=0.0):
    """
    Bob's side of the protocol. (Optimized version)
    1. Simulates Eve/Noise by intercepting/modifying the q_channel in a single batch.
    2. Generates his own random bases.
    3. Measures the received qubits (circuits) using his bases in a single batch.
    4. Publishes his bases on the c_channel.
    5. Waits for Alice to publish her bases.
    6. Sifts his key.
    7. Returns his measurement results and his final sifted key.
    """
    print(f"[Bob] Starting. (Noise/Eve level: {noise_level * 100}%)")

    # Use a single simulator instance
    simulator = AerSimulator()

    # 1. Simulate Eve/Noise
    if noise_level > 0:
        print("[Bob/Eve] Eavesdropper is intercepting the quantum channel...")

        # --- OTTIMIZZAZIONE: Preparazione del Batch di Eve ---
        intercept_info = []  # Lista per salvare (index, eve_basis)
        circuits_for_eve = []  # Lista per i circuiti da simulare

        for i in range(key_len):
            if random() < noise_level:  # Eve intercetta questo qubit
                eve_basis = choice(["Z", "X"])

                # Salviamo le info per dopo
                intercept_info.append({"index": i, "basis": eve_basis})

                # IMPORTANTE: Copiamo il circuito per non modificare l'originale
                # Altrimenti, le misurazioni di Eve "inquinano" quelle di Bob
                qc_from_alice = q_channel[i].copy()

                # Aggiungiamo le porte di misurazione di Eve
                if eve_basis == "X":
                    qc_from_alice.h(0)
                qc_from_alice.measure(0, 0)

                circuits_for_eve.append(qc_from_alice)

        # --- OTTIMIZZAZIONE: Esecuzione del Batch di Eve (una sola volta) ---
        if circuits_for_eve:  # Solo se Eve ha intercettato qualcosa
            print(
                f"[Bob/Eve] Eve is measuring {len(circuits_for_eve)} qubits in one batch..."
            )
            transpiled_eve = transpile(circuits_for_eve, simulator)
            result_eve = simulator.run(transpiled_eve, shots=1, memory=True).result()

            # Otteniamo tutti i bit misurati da Eve
            measured_bits = [
                int(result_eve.get_memory(k)[0]) for k in range(len(circuits_for_eve))
            ]

            # --- OTTIMIZZAZIONE: Ricostruzione dei qubit per Bob ---
            for k, info in enumerate(intercept_info):
                original_index = info["index"]
                eve_basis = info["basis"]
                eve_measured_bit = measured_bits[k]

                # Eve crea un *nuovo* qubit da inviare a Bob
                qc_for_bob = QuantumCircuit(1, 1)
                if eve_measured_bit == 1:
                    qc_for_bob.x(0)
                if eve_basis == "X":
                    qc_for_bob.h(0)

                # Sostituiamo il qubit originale con quello di Eve
                q_channel[original_index] = qc_for_bob

    # 2. Generate Bob's bases
    print(f"[Bob] Receiving {len(q_channel)} qubits...")
    bob_bases = [choice(["Z", "X"]) for _ in range(key_len)]

    # 3. Measure qubits (Questo era già ottimizzato in un batch)
    circuits_to_run = []
    for i in range(key_len):
        qc = q_channel[i]  # Get the (potentially tampered) qubit

        # Add Bob's measurement gates
        if bob_bases[i] == "X":
            qc.h(0)
        qc.measure(0, 0)
        circuits_to_run.append(qc)

    # Run all measurements in one batch
    transpiled_circuits = transpile(circuits_to_run, simulator)
    result = simulator.run(transpiled_circuits, shots=1, memory=True).result()

    # --- THIS IS THE FIX ---
    # We must explicitly get the memory for EACH circuit by its index
    bob_results = []
    for i in range(len(circuits_to_run)):
        # get_memory(i) returns a list (e.g., ['0'] or ['1']) since shots=1
        bob_results.append(int(result.get_memory(i)[0]))
    # --- END FIX ---

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
    Alice and Bob publicly compare a sample of their sifted keys
    to estimate the Quantum Bit Error Rate (QBER).
    """
    if not sifted_alice or not sifted_bob:
        return 0.0, [], []

    num_samples = int(len(sifted_alice) * sample_size)
    if num_samples == 0:
        print("[QBER] Sifted key is too short to sample. Aborting.")
        return 0.0, [], []

    # Randomly choose indices to compare
    sample_indices = sample(range(len(sifted_alice)), num_samples)
    sample_indices.sort(reverse=True)  # Sort to pop from the end

    print(f"[QBER] Comparing {num_samples} bits to estimate error rate...")

    errors = 0
    final_alice = list(sifted_alice)
    final_bob = list(sifted_bob)

    # For each sample, compare and *discard* the bit
    for i in sample_indices:
        alice_sample_bit = final_alice.pop(i)
        bob_sample_bit = final_bob.pop(i)
        if alice_sample_bit != bob_sample_bit:
            errors += 1

    # Gestione divisione per zero se num_samples è 0 (anche se già gestito sopra)
    if num_samples == 0:
        return 0.0, final_alice, final_bob

    qber = errors / num_samples
    return qber, final_alice, final_bob


def run_simulation(key_len, noise_level):
    """
    Runs a single simulation of the BB84 protocol using threads
    for Alice and Bob to "talk".
    """

    q_channel = []  # Represents the quantum channel
    c_channel = {}  # Represents the public classical channel

    # We use a dict to store results from the threads
    results = {}

    # Create threads
    alice_thread = threading.Thread(
        target=lambda: results.update({"alice": alice(q_channel, c_channel, key_len)})
    )
    bob_thread = threading.Thread(
        target=lambda: results.update(
            {"bob": bob(q_channel, c_channel, key_len, noise_level)}
        )
    )

    # Start threads
    alice_thread.start()
    time.sleep(0.1)  # Give Alice a tiny head start to prepare qubits
    bob_thread.start()

    # Wait for them to finish
    alice_thread.join()
    bob_thread.join()

    # --- Protocol Finished. Now analyze. ---

    # Gestione di un raro caso in cui un thread potrebbe non aver restituito risultati
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

    # Estimate QBER
    qber, final_alice, final_bob = estimate_qber(sifted_alice_key, sifted_bob_key)

    print("\n--- Final Analysis ---")
    print(f"Initial Key Length:    {key_len}")
    print(f"Sifted Key Length:     {len(sifted_alice_key)}")
    print(f"Final Key Length:      {len(final_alice)}")
    print(f"Quantum Bit Error Rate (QBER): {qber:.2%}")

    # This is the security guarantee
    if qber > 0.15:  # Theoretical max is 25%, but we set a lower security threshold
        print("!! QBER IS TOO HIGH! EAVESDROPPING DETECTED. KEY DISCARDED. !!")
        return qber, 0  # Return QBER and final key length
    else:
        print(">> QBER is low. Secure key established.")
        print(f">> Final Shared Key:  {''.join(map(str, final_alice))}")
        print(f">> Keys Match:        {final_alice == final_bob}")
        return qber, len(final_alice)


# Main execution
def main():
    KEY_LENGTH = 40  # Number of qubits to send

    # --- Scenario 1: Secure Channel ---
    print("=================================================")
    print(f" SCENARIO 1: Secure Channel (Noise = 0.0) ")
    print("=================================================")
    run_simulation(key_len=KEY_LENGTH, noise_level=0.0)

    # --- Scenario 2: Full Eavesdropping ---
    print("\n=================================================")
    print(f" SCENARIO 2: Eve Intercepts All (Noise = 1.0) ")
    print("=================================================")
    run_simulation(key_len=KEY_LENGTH, noise_level=1.0)

    # --- Scenario 3: Plotting QBER vs. Noise ---
    print("\n=================================================")
    print(" SCENARIO 3: Plotting QBER vs. Noise Level ")
    print("=================================================")
    print("Running multiple simulations... (This may take a moment)")

    noise_levels = np.linspace(0, 1, 20)
    avg_qbers = []

    for noise in noise_levels:
        print(f"Simulating noise level: {noise:.2f}")
        qbers = []
        for _ in range(5):  # Average over 5 runs for smoother data
            qber, _ = run_simulation(key_len=KEY_LENGTH, noise_level=noise)
            qbers.append(qber)
        avg_qbers.append(np.mean(qbers))

    # Plotting

    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, avg_qbers, "bo-", label="Simulated QBER")
    plt.plot(
        noise_levels,
        noise_levels * 0.25,
        "r--",
        label="Theoretical QBER (Noise * 0.25)",
    )

    plt.title("BB84: QBER vs. Eve Interception Probability")
    plt.xlabel("Eve Interception Probability (Noise Level)")
    plt.ylabel("Quantum Bit Error Rate (QBER)")
    plt.legend()
    plt.grid(True)

    # Salva il grafico invece di tentare di mostrarlo
    filename = "bb84_qber_vs_noise.png"
    plt.savefig(filename)
    print(f"\n[Plotting] Grafico salvato come: {filename}")


if __name__ == "__main__":
    main()
