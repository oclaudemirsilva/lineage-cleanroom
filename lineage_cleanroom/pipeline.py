"""cleanroom/pipeline.py — core reusável do "catch-the-leak" (SOLID / DIP).

Não conhece o dataset concreto, o classificador concreto, nem COMO a telemetria é escrita:
tudo entra por INJEÇÃO (model_factory, split fns, emit). Isso mantém o core testável, com baixo
acoplamento e aberto a extensão (OCP): trocar KNN por RandomForest, ou o split, ou o sink de
telemetria, não toca este arquivo.

Fluxo:
  dataset → features → [split ingênuo → treina → AUC inflada] → detecta vazamento →
  [split limpo group-aware → treina → AUC honesta] → manifesto assinado + verificação de adulteração.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from .datagen import Dataset
from .ingest import SplitView
from .leakage import detect_leakage, sequences_to_kmer_features
from .manifest import build_manifest, dataset_fingerprint, sign_manifest, verify_manifest
from .provenance import ProvenancePolicy, audit_label_provenance
from .telemetry import Emit, null_sink

# Tipos das dependências injetadas (DIP).
ModelFactory = Callable[[], object]                       # () -> classificador sklearn-like
SplitFn = Callable[[np.ndarray, np.ndarray], tuple]        # (y, groups) -> (idx_train, idx_test)


def _fit_eval(model_factory: ModelFactory, X, y, idx_tr, idx_te) -> float:
    clf = model_factory()
    clf.fit(X[idx_tr], y[idx_tr])
    proba = clf.predict_proba(X[idx_te])[:, 1]
    return float(roc_auc_score(y[idx_te], proba))


def run_catch_the_leak(
    dataset: Dataset,
    *,
    model_factory: ModelFactory,
    naive_split_fn: SplitFn,
    clean_split_fn: SplitFn,
    emit: "Emit | None" = None,
) -> dict:
    emit = emit or null_sink()

    X = sequences_to_kmer_features(dataset.seqs)
    y, groups = dataset.y, dataset.groups
    emit("dataset_built", {"n_rows": int(len(y)), "n_features": int(X.shape[1]),
                           "pos_rate": round(float(y.mean()), 3)})

    # 1) Split INGÊNUO (por linha) → vazamento silencioso → AUC inflada.
    tr, te = naive_split_fn(y, groups)
    auc_naive = _fit_eval(model_factory, X, y, tr, te)
    leak = detect_leakage(X[tr], X[te], groups[tr], groups[te])
    emit("naive_eval", {"auc_naive": round(auc_naive, 4)})
    emit("leak_detected", leak)

    # 2) Split LIMPO (group-aware) → sem vazamento → AUC honesta.
    tr2, te2 = clean_split_fn(y, groups)
    auc_clean = _fit_eval(model_factory, X, y, tr2, te2)
    leak_clean = detect_leakage(X[tr2], X[te2], groups[tr2], groups[te2])
    emit("clean_eval", {"auc_clean": round(auc_clean, 4),
                        "residual_leak_fraction": leak_clean["test_leak_fraction"]})

    inflation = auc_naive - auc_clean
    verdict = (
        "LEAK DETECTED — naive metric inflated by train/test leakage; clean metric is honest"
        if leak["leaked"] and inflation > 0.02
        else "no material leakage detected"
    )

    # 3) Manifesto assinado (tamper-evident) + prova de adulteração.
    manifest = build_manifest(
        dataset_fp=dataset_fingerprint(X, y),
        split="naive(random-by-row) -> clean(group-aware)",
        leak_report=leak,
        metrics={"auc_naive": round(auc_naive, 4), "auc_clean": round(auc_clean, 4),
                 "inflation": round(inflation, 4)},
        verdict=verdict,
    )
    signed, _ = sign_manifest(manifest)
    verify_ok = verify_manifest(signed)

    tampered = dict(signed)
    tampered["metrics"] = dict(tampered["metrics"], auc_clean=0.99)  # forja a métrica honesta
    verify_after_tamper = verify_manifest(tampered)
    emit("manifest_signed", {"verify_ok": verify_ok, "verify_after_tamper": verify_after_tamper})

    return {
        "auc_naive": auc_naive,
        "auc_clean": auc_clean,
        "inflation": inflation,
        "leak_report": leak,
        "leak_report_clean": leak_clean,
        "verdict": verdict,
        "manifest": signed,
        "verify_ok": verify_ok,
        "verify_after_tamper": verify_after_tamper,
    }


@dataclass(frozen=True)
class AuditResult:
    """Resultado da auditoria de um split REAL (o produto). `passed=True` = sem contaminação."""
    passed: bool
    verdict: str
    leak_report: dict
    provenance_report: dict
    manifest: dict          # assinado
    verify_ok: bool
    n_train: int
    n_test: int


def audit_split(
    view: SplitView,
    *,
    policy: "Optional[ProvenancePolicy]" = None,
    emit: "Emit | None" = None,
) -> AuditResult:
    """Audita um split treino/teste JÁ EXISTENTE — o caso real do pesquisador ("eu já separei,
    está vazando?"). NÃO treina modelo: roda o gate de vazamento (features) + o gate de
    proveniência de rótulo (origem do y), e emite um manifesto assinado. Estático, determinístico."""
    emit = emit or null_sink()
    policy = policy or ProvenancePolicy()

    emit("audit_start", {"n_train": view.n_train, "n_test": view.n_test,
                         "n_features": len(view.feature_names),
                         "has_groups": view.groups_train is not None,
                         "has_provenance": view.prov_train is not None})

    # Gate 1 — vazamento (olha as features).
    leak = detect_leakage(view.X_train, view.X_test, view.groups_train, view.groups_test)
    emit("leak_detected", leak)

    # Gate 2 — proveniência de rótulo (olha a origem do y).
    prov = audit_label_provenance(view.prov_train, view.prov_test, policy)
    emit("provenance_audited", {"provided": prov.get("provided"),
                                "violations": len(prov.get("violations", []))})

    contaminated = bool(leak["leaked"]) or bool(prov.get("violations"))
    passed = not contaminated
    verdict = _audit_verdict(leak, prov)

    X_all = np.vstack([view.X_train, view.X_test])
    y_all = np.concatenate([view.y_train, view.y_test])
    manifest = build_manifest(
        dataset_fp=dataset_fingerprint(X_all, y_all),
        split="user-provided train/test",
        leak_report=leak,
        provenance_report=prov,
        metrics={"n_train": view.n_train, "n_test": view.n_test},
        verdict=verdict,
    )
    signed, _ = sign_manifest(manifest)
    verify_ok = verify_manifest(signed)
    emit("manifest_signed", {"verify_ok": verify_ok, "passed": passed})

    return AuditResult(
        passed=passed, verdict=verdict, leak_report=leak, provenance_report=prov,
        manifest=signed, verify_ok=verify_ok, n_train=view.n_train, n_test=view.n_test,
    )


def _audit_verdict(leak: dict, prov: dict) -> str:
    parts = []
    if leak["leaked"]:
        frac = leak["test_leak_fraction"] * 100
        parts.append(f"LEAKAGE - {frac:.1f}% of test rows overlap training")
    if prov.get("violations"):
        parts.append("PROVENANCE - " + "; ".join(prov["violations"]))
    if not parts:
        note = "" if prov.get("provided") else " (label provenance not provided - not audited)"
        return "CLEAN - no train/test leakage or provenance violation detected" + note
    return "CONTAMINATION DETECTED - " + " | ".join(parts)
