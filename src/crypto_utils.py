# crypto_utils.py
from __future__ import annotations
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

NONCE_SIZE = 12  # GCM standard

def encrypt_gcm(plaintext: bytes, key: bytes, aad: bytes | None = None) -> bytes:
    nonce = os.urandom(NONCE_SIZE)
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
    enc = cipher.encryptor()
    if aad:
        enc.authenticate_additional_data(aad)
    ct = enc.update(plaintext) + enc.finalize()
    return nonce + ct + enc.tag

def decrypt_gcm(blob: bytes, key: bytes, aad: bytes | None = None) -> bytes:
    if len(blob) < NONCE_SIZE + 16:
        raise ValueError("Ciphertext too short")
    nonce = blob[:NONCE_SIZE]
    tag = blob[-16:]
    ct = blob[NONCE_SIZE:-16]
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
    dec = cipher.decryptor()
    if aad:
        dec.authenticate_additional_data(aad)
    return dec.update(ct) + dec.finalize()
