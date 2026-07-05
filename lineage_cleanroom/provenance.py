"""lineage_cleanroom/provenance.py — gate de PROVENIÊNCIA de rótulo (SRP: só julga a origem do y).

Diferente do gate de vazamento (que olha as FEATURES), este olha de ONDE veio o RÓTULO. Cada
linha pode carregar uma etiqueta de origem: "human" (atestado por mão humana), "model:..."
(gerado por um modelo), "heuristic", "unknown". Duas regras de integridade, ambas configuráveis:

  1. GOLD-HUMANO: o conjunto de avaliação (o "gabarito") deve ser 100% atestado por humano — senão
     você mede um modelo contra rótulos que não são verdade.
  2. ANTI-AUTOFAGIA: o treino não pode usar rótulos gerados pelo PRÓPRIO tipo de modelo como se
     fossem verdade (o modelo aprende da própria saída e a métrica vira ilusão).

Reusa o princípio do corpus ("gold = só mão humana", canário humano). Fail-closed: se a política
exige humano e há não-humano, o gate REPROVA.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

# Prefixos canônicos de origem (case-insensitive). Valores livres no dado são normalizados por prefixo.
HUMAN = "human"
MODEL = "model"
HEURISTIC = "heuristic"
UNKNOWN = "unknown"


def normalize_source(tag: object) -> str:
    """Normaliza uma etiqueta livre para uma classe canônica pelo prefixo ('model:v2' -> 'model')."""
    s = str(tag).strip().lower()
    for canon in (HUMAN, MODEL, HEURISTIC):
        if s.startswith(canon):
            return canon
    return UNKNOWN


def is_human(tag: object) -> bool:
    return normalize_source(tag) == HUMAN


@dataclass(frozen=True)
class ProvenancePolicy:
    """Política de integridade de rótulo. Defaults conservadores (fail-closed no gold)."""
    require_human_gold: bool = True          # avaliação só com rótulo humano
    forbid_model_labels_in_train: bool = False  # anti-autofagia opcional no treino
    forbidden_train_sources: frozenset = field(default_factory=lambda: frozenset({MODEL}))


def audit_label_provenance(
    prov_train: "Optional[Sequence]",
    prov_test: "Optional[Sequence]",
    policy: "Optional[ProvenancePolicy]" = None,
) -> dict:
    """Relatório serializável da origem dos rótulos + violações da política. Se a proveniência não
    foi fornecida, marca `provided=False` e NÃO inventa veredito (honesto)."""
    policy = policy or ProvenancePolicy()

    if prov_test is None and prov_train is None:
        return {"provided": False, "note": "coluna de proveniência não fornecida — origem do rótulo não auditada",
                "violations": [], "leaked": False, "passed": None}

    dist_train = _dist(prov_train)
    dist_test = _dist(prov_test)
    violations: list[str] = []

    # Rule 1: human-attested gold in the evaluation (test) set.
    if policy.require_human_gold and prov_test is not None:
        non_human = sum(v for k, v in dist_test.items() if k != HUMAN)
        if non_human > 0:
            violations.append(
                f"NON-HUMAN GOLD: {non_human} evaluation label(s) are not human-attested "
                f"(distribution: {dist_test})")

    # Rule 2: anti-autophagy in the training set (optional).
    if policy.forbid_model_labels_in_train and prov_train is not None:
        bad = sum(v for k, v in dist_train.items() if k in policy.forbidden_train_sources)
        if bad > 0:
            violations.append(
                f"AUTOPHAGY: {bad} training label(s) from forbidden source(s) "
                f"{set(policy.forbidden_train_sources)} used as ground truth")

    human_frac_test = (dist_test.get(HUMAN, 0) / max(1, sum(dist_test.values()))) if prov_test is not None else None

    return {
        "provided": True,
        "dist_train": dist_train,
        "dist_test": dist_test,
        "human_fraction_test": round(human_frac_test, 4) if human_frac_test is not None else None,
        "violations": violations,
        "leaked": len(violations) > 0,   # "leaked" aqui = contaminação de proveniência
        "passed": len(violations) == 0,
        "policy": {
            "require_human_gold": policy.require_human_gold,
            "forbid_model_labels_in_train": policy.forbid_model_labels_in_train,
        },
    }


def _dist(prov: "Optional[Sequence]") -> dict:
    if prov is None:
        return {}
    return dict(Counter(normalize_source(t) for t in prov))
