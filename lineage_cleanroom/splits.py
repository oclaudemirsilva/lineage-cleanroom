"""cleanroom/splits.py — estratégias de split injetáveis (OCP: adicione novas sem tocar o core).

`naive_random_split` = o split que INTRODUZ vazamento (aleatório por linha, ignora grupos).
`group_aware_split`  = o split LIMPO (nenhum grupo cruza a fronteira treino/teste).
Ambos casam a assinatura `SplitFn = (y, groups) -> (idx_train, idx_test)`.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_naive_random_split(test_size: float = 0.3, seed: int = 0):
    def split(y: np.ndarray, groups: np.ndarray):
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
        return tr, te
    return split


def make_group_aware_split(test_size: float = 0.3, seed: int = 0):
    def split(y: np.ndarray, groups: np.ndarray):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(np.arange(len(y)), y, groups))
        return tr, te
    return split
