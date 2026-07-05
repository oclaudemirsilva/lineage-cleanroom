"""lineage_cleanroom/manifest.py — manifesto de proveniência assinado (Ed25519), tamper-evident.

O manifesto registra o LINEAGE (fingerprint do dataset, descrição do split, relatório de
vazamento, métricas antes/depois) e é assinado com Ed25519. Qualquer alteração de um único
byte do conteúdo invalida a assinatura → prova de adulteração. Serialização canônica
(JSON sort_keys, separadores fixos) para que assinatura e verificação sejam determinísticas.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def dataset_fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    """SHA-256 do conteúdo (X, y). Reproduzível; muda se um valor mudar."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.round(X, 8)).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_manifest(
    *,
    dataset_fp: str,
    split: str,
    leak_report: dict,
    metrics: dict,
    verdict: str,
    provenance_report: "Optional[dict]" = None,
    tool: str = "Lineage CleanRoom",
    version: str = "0.1.0",
) -> dict:
    """Monta o payload (ainda NÃO assinado)."""
    return {
        "tool": tool,
        "version": version,
        "dataset_fingerprint": dataset_fp,
        "split": split,
        "leak_report": leak_report,
        "provenance_report": provenance_report or {},
        "metrics": metrics,
        "verdict": verdict,
    }


def sign_manifest(
    manifest: dict, private_key: "Optional[Ed25519PrivateKey]" = None
) -> tuple[dict, Ed25519PrivateKey]:
    """Assina o payload canônico. Devolve (dict_assinado, chave_privada).
    O dict_assinado embute a chave pública (hex) e a assinatura (hex)."""
    if private_key is None:
        private_key = Ed25519PrivateKey.generate()
    msg = _canonical(manifest)
    sig = private_key.sign(msg)
    pub = private_key.public_key().public_bytes_raw()
    signed = dict(manifest)
    signed["_signature"] = sig.hex()
    signed["_public_key"] = pub.hex()
    return signed, private_key


def verify_manifest(signed: dict) -> bool:
    """True se a assinatura confere com o conteúdo. Falha (False) se qualquer campo mudou."""
    signed = dict(signed)
    sig_hex = signed.pop("_signature", None)
    pub_hex = signed.pop("_public_key", None)
    if not sig_hex or not pub_hex:
        return False
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
        pub.verify(bytes.fromhex(sig_hex), _canonical(signed))
        return True
    except (InvalidSignature, ValueError):
        return False
