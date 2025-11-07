# server.py
from __future__ import annotations
import binascii
from typing import Dict

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room

from network import build_network
from crypto_utils import encrypt_gcm, decrypt_gcm

app = Flask(__name__)
# eventlet semplifica il websocket server
socketio = SocketIO(app, cors_allowed_origins="*")

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

# ----------------------- main -----------------------

if __name__ == "__main__":
    # ssl_context='adhoc' genera un certificato self-signed temporaneo (solo demo!)
    socketio.run(app, port=5000, ssl_context="adhoc")
