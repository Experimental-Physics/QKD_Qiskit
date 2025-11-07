# server.py
from __future__ import annotations
import asyncio
import binascii
import os
import time
from random import choice, randint, random, sample
from typing import Dict, List, Optional, Annotated

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# =========================
# QKD / Crypto / Network imports (come nel tuo progetto)
# =========================
from network import build_network, QKDNode, NetworkState
from crypto_utils import encrypt_gcm, decrypt_gcm
from qkd import bb84_key_generation, e91_key_generation

# =========================
# Qiskit Aer — per replicare esattamente lo Scenario 4 del main
# =========================
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# =========================
# FastAPI setup
# =========================
app = FastAPI(title="QKD Playground API", version="1.1.0", description="""
Agnostic backend for QKD demos.
- REST for data/sweeps/management
- WebSocket for chat & live protocol traces
- Sweep QBER vs P(Eve) identical to 'main' Scenario 4 (with hardware noise + theory)
""")

# CORS (aperto by default; restringi in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =========================
# Global in-memory state
# =========================
STATE: NetworkState = build_network(num_nodes=2)

# =========================
# WebSocket connection manager
# =========================
class ConnectionManager:
    def __init__(self):
        self.rooms: Dict[str, List[WebSocket]] = {}

    async def connect(self, ws: WebSocket, room: str):
        await ws.accept()
        self.rooms.setdefault(room, []).append(ws)

    def _prune(self, room: str):
        self.rooms[room] = [w for w in self.rooms.get(room, []) if not w.client_state.name == "DISCONNECTED"]

    async def disconnect(self, ws: WebSocket, room: str):
        try:
            self.rooms.get(room, []).remove(ws)
        except ValueError:
            pass
        self._prune(room)

    async def send_room(self, room: str, payload: dict):
        for ws in list(self.rooms.get(room, [])):
            try:
                await ws.send_json(payload)
            except Exception:
                await self.disconnect(ws, room)

    async def broadcast(self, payload: dict):
        for room in list(self.rooms.keys()):
            await self.send_room(room, payload)

manager = ConnectionManager()

# =========================
# Pydantic models & types
# =========================
IntGE0 = Annotated[int, Field(ge=0)]
IntGE2 = Annotated[int, Field(ge=2)]
IntGE2LE400 = Annotated[int, Field(ge=2, le=400)]
IntBits = Annotated[int, Field(ge=8, le=8192)]
IntKeyLen = Annotated[int, Field(ge=8, le=4096)]

FloatGE0 = Annotated[float, Field(ge=0.0)]
FloatGE0LE1 = Annotated[float, Field(ge=0.0, le=1.0)]
FloatQBER = Annotated[float, Field(ge=0.0, le=0.5)]
FloatGT0LE09 = Annotated[float, Field(gt=0.0, le=0.9)]

class RebuildReq(BaseModel):
    num_nodes: IntGE2 = 4
    noise_min: FloatGE0 = 0.0
    noise_max: FloatGE0 = 0.05
    eve_prob: FloatGE0LE1 = 0.0
    protocol: str = Field("bb84", description="bb84|e91|mixed")

class SweepParams(BaseModel):
    # X axis = P(Eve) (identico al main Scenario 4)
    noise_min: FloatGE0 = 0.0
    noise_max: FloatGE0LE1 = 1.0
    num: IntGE2LE400 = 15
    n_bits: IntBits = 100               # KEY_LENGTH usato nel main per statistiche migliori
    protocol: str = "bb84"

    # nuovi parametri per Scenario 4
    hardware_noise: FloatGE0LE1 = 0.03  # prob. depolarizzante sui gate {X,H}
    runs: IntGE2 = 3                    # media su N run per ogni punto
    include_theory: bool = True         # curva teorica 0.25*P(Eve)
    qber_cut: FloatQBER = 0.15          # stesso threshold del main

class SweepResp(BaseModel):
    noise: list[float]                         # P(Eve)
    qber_ideal: list[float]                    # hardware 0%
    qber_hardware: list[float]                 # hardware depolarizing
    qber_theoretical: Optional[list[float]]    # 0.25*P(Eve), se richiesto
    success_ideal: list[bool]                  # almeno una run ha prodotto key (post-cut)
    success_hardware: list[bool]

class EstablishReq(BaseModel):
    src: IntGE0
    dst: IntGE0
    protocol: str = "bb84"
    n_bits: IntBits = 256
    noise: FloatGE0 = 0.02
    eve_prob: FloatGE0LE1 = 0.0
    qber_threshold: FloatQBER = 0.11

class ChatSend(BaseModel):
    sender: IntGE0
    receiver: IntGE0
    message: str

class DecryptReq(BaseModel):
    owner: IntGE0
    peer: IntGE0
    blob_hex: str

class WorkbenchReq(BaseModel):
    protocol: str = Field("bb84", description="bb84|e91")
    key_len: IntKeyLen = 64
    noise: FloatGE0LE1 = 0.0
    eve_prob: FloatGE0LE1 = 0.0
    sample_frac: FloatGT0LE09 = 0.5
    seed: int | None = None

# =========================
# Helpers per replicare lo Scenario 4 (main)
# =========================

def create_hardware_noise_model(prob_depolarizing: float) -> Optional[NoiseModel]:
    """
    Stesso modello del main: errore depolarizzante su gate single-qubit {X,H}.
    """
    if prob_depolarizing <= 0.0:
        return None
    noise_model = NoiseModel()
    error_1 = depolarizing_error(prob_depolarizing, 1)
    # Applichiamo l'errore ai gate usati nella preparazione/rotazione
    noise_model.add_quantum_error(error_1, ["x", "h"], [0])
    return noise_model

def _alice(q_channel: list, c_channel: dict, key_len: int):
    bits = [randint(0, 1) for _ in range(key_len)]
    bases = [choice(["Z", "X"]) for _ in range(key_len)]
    q_channel.clear()
    for i in range(key_len):
        qc = QuantumCircuit(1, 1)
        if bits[i] == 1:
            qc.x(0)
        if bases[i] == "X":
            qc.h(0)
        q_channel.append(qc)
    # attende bob_bases e pubblica alice_bases
    while "bob_bases" not in c_channel:
        time.sleep(0.001)
    c_channel["alice_bases"] = bases
    # sift
    sifted = [bits[i] for i in range(key_len) if bases[i] == c_channel["bob_bases"][i]]
    return bits, sifted

def _bob(q_channel: list, c_channel: dict, key_len: int, eve_intercept_prob: float, noise_model: Optional[NoiseModel]):
    simulator = AerSimulator(noise_model=noise_model)

    # Eve: intercept & resend su una frazione dei qubit, misurando Z/X a caso
    if eve_intercept_prob > 0.0:
        intercept_info = []
        circuits_for_eve = []
        for i in range(key_len):
            if random() < eve_intercept_prob:
                eve_basis = choice(["Z", "X"])
                intercept_info.append((i, eve_basis))
                qc = q_channel[i].copy()
                if eve_basis == "X":
                    qc.h(0)
                qc.measure(0, 0)
                circuits_for_eve.append(qc)
        if circuits_for_eve:
            transpiled_eve = transpile(circuits_for_eve, simulator)
            result_eve = simulator.run(transpiled_eve, shots=1, memory=True).result()
            measured_bits = [int(result_eve.get_memory(k)[0]) for k in range(len(circuits_for_eve))]
            for (k, (orig_idx, eve_basis)) in enumerate(intercept_info):
                b = measured_bits[k]
                qc_for_bob = QuantumCircuit(1, 1)
                if b == 1:
                    qc_for_bob.x(0)
                if eve_basis == "X":
                    qc_for_bob.h(0)
                q_channel[orig_idx] = qc_for_bob

    # Bob sceglie le basi, misura tutto (anche mismatch di basi)
    bob_bases = [choice(["Z", "X"]) for _ in range(key_len)]
    circuits = []
    for i in range(key_len):
        qc = q_channel[i]
        if bob_bases[i] == "X":
            qc.h(0)
        qc.measure(0, 0)
        circuits.append(qc)
    transpiled = transpile(circuits, simulator)
    result = simulator.run(transpiled, shots=1, memory=True).result()
    bob_results = [int(result.get_memory(i)[0]) for i in range(len(circuits))]

    # Pubblica basi e attende Alice
    c_channel["bob_bases"] = bob_bases
    while "alice_bases" not in c_channel:
        time.sleep(0.001)
    alice_bases = c_channel["alice_bases"]

    # Sift
    sifted = [bob_results[i] for i in range(key_len) if bob_bases[i] == alice_bases[i]]
    return bob_results, sifted

def _estimate_qber(sifted_alice: List[int], sifted_bob: List[int], sample_frac: float = 0.5):
    if not sifted_alice or not sifted_bob:
        return 0.0, [], []
    n = len(sifted_alice)
    m = max(1, int(n * sample_frac))
    idx = sorted(sample(range(n), m), reverse=True)
    errors = 0
    a = list(sifted_alice)
    b = list(sifted_bob)
    for i in idx:
        if a.pop(i) != b.pop(i):
            errors += 1
    qber = errors / m
    return qber, a, b  # a,b = chiavi finali dopo rimozione campioni

def _run_simulation_once(key_len: int, eve_intercept_prob: float, noise_model: Optional[NoiseModel], qber_cut: float):
    """
    Identico nella logica al tuo main:
    - genera canali,
    - lancia alice/bob,
    - stima QBER su metà dei bit sifted,
    - scarta se qber > qber_cut (ritorna (qber, 0)), altrimenti (qber, len(final_key)).
    """
    q_channel: List[QuantumCircuit] = []
    c_channel: Dict[str, List[int] | List[str]] = {}
    results: Dict[str, tuple] = {}

    import threading, time as _t

    t_alice = threading.Thread(target=lambda: results.update({"alice": _alice(q_channel, c_channel, key_len)}))
    t_bob = threading.Thread(target=lambda: results.update({"bob": _bob(q_channel, c_channel, key_len, eve_intercept_prob, noise_model)}))

    t_alice.start()
    _t.sleep(0.05)
    t_bob.start()
    t_alice.join()
    t_bob.join()

    if "alice" not in results or "bob" not in results:
        return 0.0, 0

    _, sifted_a = results["alice"]
    _, sifted_b = results["bob"]
    qber, final_a, final_b = _estimate_qber(sifted_a, sifted_b, sample_frac=0.5)

    if qber > qber_cut:
        return qber, 0
    return qber, len(final_a)

# =========================
# Health & basic info
# =========================
@app.get("/health")
def health():
    return {"ok": True, "nodes": len(STATE.nodes)}

@app.get("/network")
def get_network():
    return {
        "num_nodes": len(STATE.nodes),
        "edges": list(STATE.graph.edges),
        "keys": {u: list(n.keys.keys()) for u, n in STATE.nodes.items()},
        "metrics": {u: n.metrics for u, n in STATE.nodes.items()},
    }

@app.get("/metrics/{node_id}")
def get_metrics(node_id: int):
    if node_id not in STATE.nodes:
        return JSONResponse({"error": "node not found"}, status_code=404)
    return STATE.nodes[node_id].metrics

# =========================
# Section 1 — Sweep QBER vs P(Eve) (identico al main Scenario 4)
# =========================
@app.post("/sweep", response_model=SweepResp)
def sweep(params: SweepParams):
    """
    X = P(Eve) in [noise_min .. noise_max]
    Y = QBER medio su 'runs' esecuzioni.
    Due curve simulate (HW ideale, HW rumoroso) + curva teorica opzionale.
    """
    # Prepara modelli di rumore UNA sola volta (come nel main)
    noise_model_ideal = create_hardware_noise_model(0.0)   # None
    noise_model_hw    = create_hardware_noise_model(params.hardware_noise)

    # griglia sull'asse X
    xs = [params.noise_min + i*(params.noise_max-params.noise_min)/(params.num-1) for i in range(params.num)]

    qber_ideal: List[float] = []
    qber_hw: List[float] = []
    success_ideal: List[bool] = []
    success_hw: List[bool] = []

    for p in xs:
        # --- Ideal HW
        q_samples = []
        succ_any = False
        for _ in range(params.runs):
            q, keylen = _run_simulation_once(
                key_len=params.n_bits,
                eve_intercept_prob=p,
                noise_model=noise_model_ideal,
                qber_cut=params.qber_cut
            )
            q_samples.append(q)
            succ_any = succ_any or (keylen > 0)
        qber_ideal.append(float(np.mean(q_samples)))
        success_ideal.append(bool(succ_any))

        # --- Noisy HW
        q_samples = []
        succ_any = False
        for _ in range(params.runs):
            q, keylen = _run_simulation_once(
                key_len=params.n_bits,
                eve_intercept_prob=p,
                noise_model=noise_model_hw,
                qber_cut=params.qber_cut
            )
            q_samples.append(q)
            succ_any = succ_any or (keylen > 0)
        qber_hw.append(float(np.mean(q_samples)))
        success_hw.append(bool(succ_any))

    qber_theory = None
    if params.include_theory and params.protocol.lower() == "bb84":
        qber_theory = [0.25 * p for p in xs]

    return SweepResp(
        noise=xs,
        qber_ideal=qber_ideal,
        qber_hardware=qber_hw,
        qber_theoretical=qber_theory,
        success_ideal=success_ideal,
        success_hardware=success_hw,
    )

# =========================
# Section 2 — Network simulator + Secure Chat
# =========================
@app.post("/rebuild")
def rebuild(req: RebuildReq):
    global STATE
    protocol_choice = req.protocol.lower()
    STATE = build_network(
        num_nodes=req.num_nodes,
        noise_range=(req.noise_min, req.noise_max),
        eve_prob=req.eve_prob,
        default_protocol=protocol_choice,
    )
    return {"ok": True, "num_nodes": req.num_nodes}

@app.post("/establish")
def establish(req: EstablishReq):
    if req.src not in STATE.nodes or req.dst not in STATE.nodes:
        return JSONResponse({"error": "unknown node"}, status_code=404)
    key, qber, proto = STATE.nodes[req.src].establish_key(
        neighbor=req.dst,
        protocol=req.protocol,
        n_bits=req.n_bits,
        noise=req.noise,
        eve_prob=req.eve_prob,
        qber_threshold=req.qber_threshold
    )
    if key is None:
        return {"ok": False, "qber": qber, "protocol": proto, "message": "QBER too high or failed"}
    return {"ok": True, "qber": qber, "protocol": proto}

@app.post("/chat/send")
async def chat_send(payload: ChatSend):
    s = payload.sender
    r = payload.receiver
    if s not in STATE.nodes or r not in STATE.nodes:
        return JSONResponse({"error": "unknown node"}, status_code=404)
    if r not in STATE.nodes[s].keys:
        return JSONResponse({"error": "no secure channel"}, status_code=400)

    key, _, _ = STATE.nodes[s].keys[r]
    blob = encrypt_gcm(payload.message.encode("utf-8"), key)
    blob_hex = binascii.hexlify(blob).decode()

    await manager.send_room(str(r), {"type": "secure_msg", "sender": s, "blob_hex": blob_hex})
    await manager.send_room(str(s), {"type": "delivered", "to": r, "blob_hex": blob_hex})
    return {"ok": True}

@app.post("/chat/decrypt")
def chat_decrypt(payload: DecryptReq):
    o = payload.owner
    p = payload.peer
    if o not in STATE.nodes or p not in STATE.nodes:
        return JSONResponse({"error": "unknown node"}, status_code=404)
    if p not in STATE.nodes[o].keys:
        return JSONResponse({"error": "no secure channel"}, status_code=400)
    key, _, _ = STATE.nodes[o].keys[p]
    try:
        pt = decrypt_gcm(binascii.unhexlify(payload.blob_hex), key)
    except Exception as e:
        return JSONResponse({"error": f"decrypt failed: {e}"}, status_code=400)
    return {"ok": True, "plaintext": pt.decode("utf-8")}

# =========================
# Section 3 — Protocol Workbench (step trace)
# =========================
@app.post("/workbench/run")
def workbench_run(req: WorkbenchReq):
    """
    Ritorna una TRACE dettagliata (adatta al frontend per visualizzazione step-by-step).
    """
    from workbench import run_bb84_trace, run_e91_trace
    if req.protocol.lower() == "e91":
        trace = run_e91_trace(key_len=req.key_len, seed=req.seed)
    else:
        trace = run_bb84_trace(
            key_len=req.key_len, noise_level=req.noise,
            eve_prob=req.eve_prob, sample_frac=req.sample_frac, seed=req.seed
        )
    return trace

# =========================
# WebSocket endpoint (generic)
# =========================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Simple protocol:
    - {"action":"join","room":"<node_id or arbitrary>"}
    - {"action":"send","room":"<room>","payload":{...}}
    """
    room = None
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")
            if action == "join":
                room = str(msg.get("room", "lobby"))
                await manager.connect(websocket, room)
                await manager.send_room(room, {"type": "joined", "room": room})
            elif action == "send":
                target = str(msg.get("room", "lobby"))
                payload = msg.get("payload", {})
                await manager.send_room(target, {"type": "event", "payload": payload})
            else:
                await websocket.send_json({"type": "error", "message": "unknown action"})
    except WebSocketDisconnect:
        if room:
            await manager.disconnect(websocket, room)
    except Exception as e:
        if room:
            await manager.send_room(room, {"type": "error", "message": str(e)})

# =========================
# Dev entry
# =========================
if __name__ == "__main__":
    # Avvio standard; per HTTPS usare --ssl-keyfile/--ssl-certfile con uvicorn.run CLI
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
