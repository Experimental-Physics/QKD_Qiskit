## Overview

This repository provides a simulation framework for a network of n-nodes HTTPS servers that incorporate quantum key distribution (QKD) for secure encryption. The framework leverages Qiskit, IBM's open-source quantum computing software development kit, to simulate quantum protocols for key generation. Specifically, it implements two prominent QKD protocols: BB84 (Bennett-Brassard 1984) and E91 (Ekert 1991, often referred to as E94 in some contexts due to typographical variations, but standardized as E91).

The project aims to demonstrate how quantum-secure keys can be integrated into classical network communication, such as HTTPS servers, to enhance security against potential quantum threats like Shor's algorithm. This is particularly useful for educational purposes, research in quantum cryptography, and prototyping quantum-enhanced network systems.

Key features:
- Simulation of quantum key distribution protocols using Qiskit.
- Network setup for multiple nodes (servers) communicating via HTTPS with QKD-encrypted keys.
- Support for noise models and error correction in quantum simulations.
- Modular design for easy extension to other QKD protocols.

## Prerequisites

To run this simulation, you'll need:
- Python 3.8 or higher.
- Qiskit (version 0.45 or later recommended for compatibility with quantum circuits and simulators).
- Additional libraries: `requests` for HTTP communication, `cryptography` for classical encryption integration, `numpy` for numerical operations, and `matplotlib` for visualizing results (optional).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Experimental-Physics/QKD_Qiskit.git
   cd QKD_Qiskit
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Unix/Mac
   venv\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not present, install the core packages manually:
   ```
   pip install qiskit qiskit-aer requests cryptography numpy matplotlib
   ```

   Note: Qiskit-Aer is used for the quantum simulator backend. For access to real quantum hardware, sign up for an IBM Quantum account and configure your API token via `qiskit_ibm_runtime`.

## Project Structure

The repository is organized as follows:

- **`src/`**: Core source code directory.
  - `qkd_protocols.py`: Implements the BB84 and E91 protocols, including key generation, sifting, and error estimation.
  - `network_simulator.py`: Handles the n-node network setup, including server initialization and key exchange.
  - `encryption_utils.py`: Integrates QKD keys with classical encryption (e.g., AES) for HTTPS communication.
  - `noise_models.py`: Defines custom noise models for simulating realistic quantum channels.

- **`examples/`**: Sample scripts and Jupyter notebooks.
  - `bb84_example.ipynb`: Jupyter notebook demonstrating a standalone BB84 simulation.
  - `e91_example.ipynb`: Jupyter notebook for E91 protocol with entanglement-based key distribution.
  - `network_demo.py`: Script to run a multi-node HTTPS server network simulation.

- **`tests/`**: Unit tests for protocols and utilities.
  - `test_bb84.py`: Tests for BB84 implementation.
  - `test_e91.py`: Tests for E91 implementation.

- **`docs/`**: Additional documentation.
  - `protocol_details.md`: In-depth explanation of BB84 and E91.

- **`requirements.txt`**: List of Python dependencies.
- **`LICENSE`**: MIT License (or specify if different).
- **`README.md`**: This file.

## Usage

### Running a Standalone Protocol Simulation

1. For BB84:
   ```
   python examples/bb84_example.py
   ```
   Or open `examples/bb84_example.ipynb` in Jupyter for an interactive session.

2. For E91:
   ```
   python examples/e91_example.py
   ```
   Similarly, use the Jupyter notebook for visualization.

These examples simulate key distribution between Alice and Bob, including potential eavesdropping (Eve) scenarios, and output metrics like quantum bit error rate (QBER).

### Running the Network Simulation

To simulate an n-node HTTPS network:

1. Configure the number of nodes and other parameters in `config.yaml` (if available) or directly in `network_simulator.py`.

2. Run the simulator:
   ```
   python src/network_simulator.py --nodes 3 --protocol bb84
   ```
   Options:
   - `--nodes N`: Number of nodes (default: 2).
   - `--protocol {bb84, e91}`: Choose the QKD protocol.
   - `--noise-level FLOAT`: Simulate channel noise (default: 0.0).
   - `--eavesdrop`: Enable eavesdropping simulation.

The script will start multiple server instances, perform QKD key exchanges, and demonstrate secure data transmission over HTTPS using the generated keys.

Example output:
```
Node 1 and Node 2: Key exchange successful. QBER: 0.02
Secure HTTPS connection established.
```

### Integrating with Real Quantum Hardware

If you have access to IBM Quantum systems:
1. Set up your IBM Quantum account:
   ```
   from qiskit_ibm_runtime import QiskitRuntimeService
   service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_API_TOKEN")
   ```
2. Modify the backend in the protocol scripts from `AerSimulator` to a real device, e.g., `service.least_busy()`.

Note: Real hardware introduces higher noise, so error correction (e.g., via Cascade or other reconciliation methods) is crucial.

## Protocols Explained

### BB84 Protocol
- **Overview**: A prepare-and-measure protocol where Alice sends polarized qubits to Bob, who measures them in random bases.
- **Steps**:
  1. Alice generates random bits and bases, encodes into qubits.
  2. Bob measures in random bases.
  3. Basis sifting and error estimation.
  4. Privacy amplification for secure key.
- **Implementation**: Uses Qiskit's `QuantumCircuit` for qubit preparation and measurement.

### E91 Protocol
- **Overview**: Entanglement-based protocol using Bell states to detect eavesdroppers via CHSH inequality violation.
- **Steps**:
  1. Source generates entangled pairs (e.g., EPR pairs).
  2. Alice and Bob measure in correlated bases.
  3. Test for entanglement violation; sift keys.
  4. Error correction and amplification.
- **Implementation**: Leverages Qiskit's entanglement circuits and violation checks.

For more details, see `docs/protocol_details.md`.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Please ensure your code follows PEP8 standards and includes tests. For major changes, open an issue first to discuss.

## Acknowledgments

- Winner of the Qiskit Fall  Fest Hackaton.
- Inspired by quantum cryptography research from Bennett, Brassard, and Ekert.
- Thanks to the open-source community for tools and tutorials on QKD simulations.

---

This README provides a complete guide to get started. If the repository evolves, update this file accordingly!
