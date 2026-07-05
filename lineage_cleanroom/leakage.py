"""cleanroom/leakage.py — detector de data leakage (vazamento treino/teste).

Duas checagens independentes, ambas determinísticas e sem rede:
  1. FEATURE-HASH: linhas de teste cujo vetor de features é idêntico (até `decimals`) a
     alguma linha de treino → memorização por duplicata. Não precisa de metadados.
  2. GROUP-AWARE: se cada linha tem um `group_id` (ex.: sequência-base, doador, batch,
     região genômica), conta grupos que cruzam a fronteira treino/teste → o vazamento
     "clássico" que um split aleatório por-linha introduz silenciosamente.

Devolve um relatório serializável (dict) — vira evidência no manifesto assinado.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np

_ALPHABET = "ACGT"


def sequences_to_kmer_features(seqs: Sequence[str], k: int = 3) -> np.ndarray:
    """Converte sequências (ACGT) em contagens de k-mers — features numéricas reproduzíveis."""
    kmers = _all_kmers(k)
    index = {km: i for i, km in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)), dtype=np.float64)
    for r, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            j = index.get(s[i:i + k])
            if j is not None:
                X[r, j] += 1.0
    return X


def _all_kmers(k: int) -> list[str]:
    out = [""]
    for _ in range(k):
        out = [p + a for p in out for a in _ALPHABET]
    return out


def _row_hashes(X: np.ndarray, decimals: int) -> list[str]:
    Xr = np.round(X, decimals)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in Xr]


def detect_leakage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    groups_train: "Optional[Sequence]" = None,
    groups_test: "Optional[Sequence]" = None,
    decimals: int = 6,
) -> dict:
    """Relatório de vazamento treino→teste. `test_leak_fraction` é a fração de linhas de
    teste que têm uma duplicata exata de features no treino."""
    train_hashes = set(_row_hashes(X_train, decimals))
    test_hashes = _row_hashes(X_test, decimals)
    leaked_rows = sum(1 for h in test_hashes if h in train_hashes)
    n_test = max(1, len(test_hashes))

    report = {
        "check": "feature_hash + group_aware",
        "decimals": decimals,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_dup_test_rows": int(leaked_rows),
        "test_leak_fraction": round(leaked_rows / n_test, 4),
        "group_span_test_rows": None,
        "leaked": leaked_rows > 0,
    }

    if groups_train is not None and groups_test is not None:
        train_groups = set(groups_train)
        span = sum(1 for g in groups_test if g in train_groups)
        report["group_span_test_rows"] = int(span)
        report["leaked"] = report["leaked"] or span > 0

    return report
