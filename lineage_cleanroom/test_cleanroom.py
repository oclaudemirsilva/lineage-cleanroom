"""lineage_cleanroom/test_cleanroom.py — prova que o firewall FUNCIONA (não é promessa).

Rode: python -m pytest lineage_cleanroom/test_cleanroom.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from lineage_cleanroom import (
    ProvenancePolicy,
    audit_label_provenance,
    audit_split,
    build_manifest,
    detect_leakage,
    load_split_from_single,
    make_group_aware_split,
    make_leaky_dna_dataset,
    make_naive_random_split,
    run_catch_the_leak,
    sign_manifest,
    verify_manifest,
    write_report,
)
from lineage_cleanroom.cli import main as cli_main

# ----------------------------- demonstrador (catch-the-leak) -----------------------------


def _result():
    ds = make_leaky_dna_dataset(seed=7)
    return run_catch_the_leak(
        ds,
        model_factory=lambda: RandomForestClassifier(n_estimators=120, random_state=0, n_jobs=-1),
        naive_split_fn=make_naive_random_split(seed=0),
        clean_split_fn=make_group_aware_split(seed=0),
    )


def test_leak_inflates_metric():
    r = _result()
    assert r["auc_naive"] > r["auc_clean"] + 0.10, (r["auc_naive"], r["auc_clean"])


def test_detector_flags_leak_and_clean_is_clean():
    r = _result()
    assert r["leak_report"]["test_leak_fraction"] > 0.30
    assert r["leak_report"]["group_span_test_rows"] > 0
    assert r["leak_report_clean"]["test_leak_fraction"] == 0.0
    assert r["verdict"].startswith("LEAK DETECTED")


def test_manifest_is_tamper_evident():
    r = _result()
    assert r["verify_ok"] is True
    assert r["verify_after_tamper"] is False


def test_detector_reports_zero_when_no_overlap():
    Xtr = np.zeros((5, 4)); Xtr[:, 0] = np.arange(5)
    Xte = np.zeros((3, 4)); Xte[:, 0] = np.arange(100, 103)
    rep = detect_leakage(Xtr, Xte)
    assert rep["feature_dup_test_rows"] == 0
    assert rep["leaked"] is False


def test_manifest_roundtrip_signature():
    m = build_manifest(dataset_fp="abc", split="x", leak_report={}, metrics={}, verdict="ok")
    signed, _ = sign_manifest(m)
    assert verify_manifest(signed) is True
    signed["verdict"] = "forged"
    assert verify_manifest(signed) is False


# ----------------------------- gate de proveniência de rótulo -----------------------------


def test_provenance_flags_nonhuman_gold():
    rep = audit_label_provenance(["human", "human"], ["human", "model:v2"], ProvenancePolicy())
    assert rep["provided"] is True
    assert rep["passed"] is False
    assert any("GOLD" in v for v in rep["violations"])


def test_provenance_passes_all_human():
    rep = audit_label_provenance(["human"], ["human", "human"], ProvenancePolicy())
    assert rep["passed"] is True
    assert rep["human_fraction_test"] == 1.0


def test_provenance_not_provided_is_honest():
    rep = audit_label_provenance(None, None, ProvenancePolicy())
    assert rep["provided"] is False
    assert rep["passed"] is None


def test_provenance_anti_autophagy_train():
    pol = ProvenancePolicy(require_human_gold=False, forbid_model_labels_in_train=True)
    rep = audit_label_provenance(["human", "model:self"], ["human"], pol)
    assert any("AUTOFAGIA" in v for v in rep["violations"])


# ----------------------------- ingestão + audit_split (produto real, via CSV) -----------------------------


def _write_csv(path, *, leak: bool, human_gold: bool = True):
    """CSV com 2 features, rótulo, grupo, proveniência e split. `leak=True` põe o MESMO grupo em
    treino e teste (vazamento)."""
    rows = []
    rng = np.random.default_rng(0)
    for g in range(20):
        y = int(g % 2)
        f0, f1 = float(g), float(rng.integers(0, 5))
        # treino
        rows.append({"f0": f0, "f1": f1, "y": y, "donor": g,
                     "src": "human", "split": "train"})
        # teste: se leak, reusa o MESMO grupo/feature; senão, grupo novo
        gt = g if leak else 100 + g
        rows.append({"f0": (f0 if leak else f0 + 1000), "f1": f1, "y": y, "donor": gt,
                     "src": ("human" if human_gold else "model:v1"), "split": "test"})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_audit_split_catches_real_leak_via_csv(tmp_path):
    csv = tmp_path / "leaky.csv"
    _write_csv(csv, leak=True)
    view = load_split_from_single(str(csv), label_col="y", split_col="split",
                                  group_col="donor", provenance_col="src")
    res = audit_split(view)
    assert res.passed is False
    assert res.leak_report["group_span_test_rows"] > 0
    assert res.verify_ok is True
    # escreve relatório em disco
    paths = write_report(res, tmp_path)
    assert (tmp_path / "lineage_cleanroom.manifest.json").exists()
    assert (tmp_path / "lineage_cleanroom.report.md").exists()


def test_audit_split_clean_dataset_passes(tmp_path):
    csv = tmp_path / "clean.csv"
    _write_csv(csv, leak=False)
    view = load_split_from_single(str(csv), label_col="y", split_col="split",
                                  group_col="donor", provenance_col="src")
    res = audit_split(view)
    assert res.passed is True
    assert res.leak_report["group_span_test_rows"] == 0
    assert res.verdict.startswith("CLEAN")


def test_audit_split_flags_nonhuman_gold(tmp_path):
    csv = tmp_path / "silvergold.csv"
    _write_csv(csv, leak=False, human_gold=False)  # teste rotulado por modelo
    view = load_split_from_single(str(csv), label_col="y", split_col="split",
                                  group_col="donor", provenance_col="src")
    res = audit_split(view)
    assert res.passed is False
    assert any("GOLD" in v for v in res.provenance_report["violations"])


def test_cli_scan_exit_code_on_leak(tmp_path):
    csv = tmp_path / "leaky.csv"
    _write_csv(csv, leak=True)
    code = cli_main(["scan", "--data", str(csv), "--label", "y", "--split-col", "split",
                     "--group", "donor", "--provenance", "src", "--out", str(tmp_path)])
    assert code == 1  # contaminação detectada
    assert (tmp_path / "lineage_cleanroom.manifest.json").exists()


def test_cli_scan_exit_code_on_clean(tmp_path):
    csv = tmp_path / "clean.csv"
    _write_csv(csv, leak=False)
    code = cli_main(["scan", "--data", str(csv), "--label", "y", "--split-col", "split",
                     "--group", "donor", "--provenance", "src"])
    assert code == 0  # limpo
