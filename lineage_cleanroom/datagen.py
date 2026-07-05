"""cleanroom/datagen.py — dataset sintético de DNA com VAZAMENTO conhecido (SRP: só gera dados).

Imita a tarefa "predição de atividade regulatória de DNA": sequências ACGT com um rótulo binário
governado por um sinal FRACO de k-mers + ruído (então o teto honesto NÃO é 1.0). O vazamento é
introduzido de propósito: o dataset amostra sequências-base COM REPOSIÇÃO (duplicatas exatas) e
mutações de 1 base (quase-duplicatas). Cada linha carrega o `group_id` = índice da sequência-base.

Um split aleatório por-LINHA espalha o mesmo grupo em treino E teste → memorização → métrica inflada.
Um split por-GRUPO (group-aware) elimina o vazamento → métrica honesta. O `group_id` é o gancho que
o detector e o split limpo usam. Nada aqui depende de biologia real — o MECANISMO do vazamento é real.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .leakage import sequences_to_kmer_features

_ALPHABET = "ACGT"


@dataclass(frozen=True)
class Dataset:
    seqs: list[str]
    y: np.ndarray          # rótulo binário por linha
    groups: np.ndarray     # group_id (sequência-base) por linha


def make_leaky_dna_dataset(
    *,
    seed: int = 7,
    n_base: int = 400,
    seq_len: int = 50,
    n_rows: int = 2000,
    dup_fraction: float = 0.6,
    label_noise: float = 0.12,
) -> Dataset:
    rng = np.random.default_rng(seed)

    base = ["".join(rng.choice(list(_ALPHABET), size=seq_len)) for _ in range(n_base)]

    # Sinal MODERADO e REAL: rótulo da sequência-base = score linear sobre o vetor de k-mers
    # (função do padrão inteiro, não de um único motivo) acima da mediana, com ruído. Assim o
    # modelo HONESTO recupera parte do sinal em grupos novos (AUC bem acima do acaso), enquanto
    # o vazamento por duplicata é o que infla a AUC ingênua.
    Xb = sequences_to_kmer_features(base)
    w = rng.normal(size=Xb.shape[1])
    raw = Xb @ w + rng.normal(0, 3.0, size=n_base)
    base_label = (raw > np.median(raw)).astype(np.int64)
    flip = rng.random(n_base) < label_noise
    base_label = np.where(flip, 1 - base_label, base_label)

    seqs: list[str] = []
    y: list[int] = []
    groups: list[int] = []
    for _ in range(n_rows):
        g = int(rng.integers(0, n_base))
        s = base[g]
        if rng.random() >= dup_fraction:  # quase-duplicata: muta 1 posição
            pos = int(rng.integers(0, seq_len))
            s = s[:pos] + rng.choice(list(_ALPHABET)) + s[pos + 1:]
        seqs.append(s)
        y.append(int(base_label[g]))
        groups.append(g)

    return Dataset(seqs=seqs, y=np.array(y), groups=np.array(groups))
