# server.py
from __future__ import annotations
import binascii
from typing import Dict

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room

from network import build_network
from crypto_utils import encrypt_gcm, decrypt_gcm
import os
from flask import Flask, request, jsonify
from qkd import bb84_key_generation


app = Flask(__name__)
# eventlet semplifica il websocket server
socketio = SocketIO(app, cors_allowed_origins="*")

async_mode = os.getenv("ASYNC_MODE", "threading")  # "threading" or "eventlet"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

# Crea rete demo (4 nodi completi)
G, nodes = build_network(num_nodes=4)

# ----------------------- REST: metriche -----------------------

@app.get("/metrics/<int:node_id>")
def get_metrics(node_id: int):
    if node_id not in nodes:
        return jsonify({"error": "node not found"}), 404
    return jsonify(nodes[node_id].metrics)

# ----------------------- Socket.IO: chat sicura -----------------------

# I client devono chiamare questo per entrare nella "stanza" del proprio ID
@socketio.on("join")
def on_join(data):
    """
    data = {"node_id": int}
    """
    node_id = int(data.get("node_id", -1))
    if node_id not in nodes:
        emit("error", {"msg": "Unknown node_id"})
        return
    join_room(str(node_id))
    emit("joined", {"room": str(node_id)})

@socketio.on("message")
def handle_message(data):
    """
    data = {"sender": int, "receiver": int, "msg": str}
    """
    try:
        sender = int(data["sender"])
        receiver = int(data["receiver"])
        msg = str(data["msg"])
    except Exception:
        emit("error", {"msg": "Invalid payload"})
        return

    if receiver not in nodes[sender].keys:
        emit("error", {"msg": "No secure channel with receiver"})
        return

    key, _, _ = nodes[sender].keys[receiver]
    blob = encrypt_gcm(msg.encode("utf-8"), key)
    # invio nella stanza del receiver
    emit("secure_msg",
         {"sender": sender, "blob_hex": binascii.hexlify(blob).decode()},
         room=str(receiver))

@socketio.on("decrypt")
def handle_decrypt(data):
    """
    data = {"owner": int, "peer": int, "blob_hex": str}
    Il client può chiedere di decifrare (per test/UX): il server verifica di avere la chiave
    dal punto di vista di 'owner' verso 'peer', e restituisce il plaintext.
    """
    try:
        owner = int(data["owner"])
        peer = int(data["peer"])
        blob = binascii.unhexlify(data["blob_hex"])
    except Exception:
        emit("error", {"msg": "Invalid payload"})
        return

    if peer not in nodes[owner].keys:
        emit("error", {"msg": "No secure channel"})
        return

    key, _, _ = nodes[owner].keys[peer]
    try:
        pt = decrypt_gcm(blob, key)
    except Exception as e:
        emit("error", {"msg": f"Decryption failed: {e}"})
        return

    emit("plaintext", {"owner": owner, "peer": peer, "text": pt.decode("utf-8")})


# ----------------------- Admin: rebuild network -----------------------
@app.post("/rebuild")
def rebuild_network():
    """
    Rebuild the QKD network with a new number of nodes (and optional noise/eve params).
    Body JSON: {"num_nodes": 6, "noise_range":[0.0,0.05], "eve_prob":0.0}
    """
    data = request.get_json(silent=True) or {}
    num_nodes  = int(data.get("num_nodes", 4))
    noise_rng  = data.get("noise_range", [0.0, 0.05])
    eve_prob   = float(data.get("eve_prob", 0.0))
    if not (isinstance(noise_rng, (list, tuple)) and len(noise_rng) == 2):
        return jsonify({"error": "noise_range must be [min,max]"}), 400

    global G, nodes
    G, nodes = build_network(num_nodes=num_nodes,
                             noise_range=(float(noise_rng[0]), float(noise_rng[1])),
                             eve_prob=eve_prob)
    return jsonify({"ok": True, "num_nodes": num_nodes})


# ----------------------- Analysis: QBER sweep vs noise -----------------------
@app.get("/sweep_bb84")
def sweep_bb84():
    """
    Run a BB84 sweep over noise ∈ [noise_min, noise_max], sampled 'num' points.
    Query args: ?noise_min=0.0&noise_max=0.12&num=25&n_bits=256&eve=0
    Returns: {"noise":[...], "qber":[...], "success":[...]}
    """
    try:
        noise_min = float(request.args.get("noise_min", 0.0))
        noise_max = float(request.args.get("noise_max", 0.12))
        num       = int(request.args.get("num", 25))
        n_bits    = int(request.args.get("n_bits", 256))
        eve       = request.args.get("eve", "0") in ("1", "true", "True")
    except Exception as e:
        return jsonify({"error": f"bad params: {e}"}), 400

    if num < 2:
        return jsonify({"error": "num must be >= 2"}), 400

    xs = [noise_min + i*(noise_max-noise_min)/(num-1) for i in range(num)]
    qbers = []
    succ  = []

    for p in xs:
        key, qber = bb84_key_generation(n_bits=n_bits,
                                        noise_level=p,
                                        eve=eve,
                                        raise_on_high_qber=False)
        qbers.append(qber)
        succ.append(bool(key))

    return jsonify({"noise": xs, "qber": qbers, "success": succ})

# ----------------------- main -----------------------

if __name__ == "__main__":
    if async_mode == "eventlet":
        # HTTP only unless you pass cert/key (see Option A)
        socketio.run(app, host="0.0.0.0", port=5000)
    else:
        # Werkzeug supports ssl_context
        socketio.run(app, host="0.0.0.0", port=5000, ssl_context="adhoc")
