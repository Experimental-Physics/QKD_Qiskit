# server.py
from __future__ import annotations
import asyncio
import binascii
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conint, confloat
import uvicorn

from network import build_network, QKDNode, NetworkState
from crypto_utils import encrypt_gcm, decrypt_gcm
from qkd import bb84_key_generation, e91_key_generation

# =========================
# FastAPI setup
# =========================
app = FastAPI(title="QKD Playground API", version="1.0.0", description="""
Agnostic backend for QKD demos.
- REST for data/sweeps/management
- WebSocket for chat & live protocol traces
""")

# CORS wide open by default (restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =========================
# Global in-memory state
# =========================
STATE: NetworkState = build_network(num_nodes=4)

# =========================
# WebSocket connection manager
# =========================
class ConnectionManager:
    def __init__(self):
        # room per node_id -> set of websockets
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
# Pydantic models
# =========================
class RebuildReq(BaseModel):
    num_nodes: conint(ge=2) = 4
    noise_min: confloat(ge=0.0) = 0.0
    noise_max: confloat(ge=0.0) = 0.05
    eve_prob: confloat(ge=0.0, le=1.0) = 0.0
    protocol: str = Field("bb84", description="bb84|e91|mixed")

class SweepResp(BaseModel):
    noise: List[float]
    qber: List[float]
    success: List[bool]

class SweepParams(BaseModel):
    noise_min: confloat(ge=0.0) = 0.0
    noise_max: confloat(ge=0.0) = 0.12
    num: conint(ge=2, le=400) = 25
    n_bits: conint(ge=8, le=8192) = 256
    eve: bool = False
    protocol: str = "bb84"

class EstablishReq(BaseModel):
    src: conint(ge=0)
    dst: conint(ge=0)
    protocol: str = "bb84"
    n_bits: conint(ge=8, le=8192) = 256
    noise: confloat(ge=0.0) = 0.02
    eve_prob: confloat(ge=0.0, le=1.0) = 0.0
    qber_threshold: confloat(ge=0.0, le=0.5) = 0.11

class ChatSend(BaseModel):
    sender: conint(ge=0)
    receiver: conint(ge=0)
    message: str

class DecryptReq(BaseModel):
    owner: conint(ge=0)
    peer: conint(ge=0)
    blob_hex: str

class WorkbenchReq(BaseModel):
    protocol: str = Field("bb84", description="bb84|e91")
    key_len: conint(ge=8, le=4096) = 64
    noise: confloat(ge=0.0, le=1.0) = 0.0
    eve_prob: confloat(ge=0.0, le=1.0) = 0.0
    sample_frac: confloat(gt=0.0, le=0.9) = 0.5
    seed: Optional[int] = None

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
# Section 1 — Sweep QBER vs Noise
# =========================
@app.post("/sweep", response_model=SweepResp)
def sweep(params: SweepParams):
    xs = [params.noise_min + i*(params.noise_max-params.noise_min)/(params.num-1) for i in range(params.num)]
    qbers, succ = [], []

    for p in xs:
        if params.protocol.lower() == "e91":
            key, qber = e91_key_generation(n_bits=params.n_bits)
        else:
            key, qber = bb84_key_generation(
                n_bits=params.n_bits, noise_level=p, eve=params.eve, raise_on_high_qber=False
            )
        qbers.append(qber)
        succ.append(bool(key))

    return SweepResp(noise=xs, qber=qbers, success=succ)

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

    # push to receiver room over WS
    await manager.send_room(str(r), {
        "type": "secure_msg",
        "sender": s,
        "blob_hex": blob_hex
    })
    # also echo to sender room as delivery confirmation
    await manager.send_room(str(s), {
        "type": "delivered",
        "to": r,
        "blob_hex": blob_hex
    })
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
    Ritorna una TRACE dettagliata (adatta a un timeline viewer nel frontend):
    - alice_bits, alice_bases, bob_bases
    - bob_results
    - sift_mask indices
    - sample_indices per stima QBER
    - qber, final_key_bits (post sample removal)
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
    Client protocol (simple JSON):
    - {"action":"join","room":"<node_id or arbitrary>"}
    - {"action":"send","room":"<room>","payload":{...}}  # echo to all in room
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
    # HTTPS dev: pass --ssl-keyfile/--ssl-certfile to uvicorn if needed
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
