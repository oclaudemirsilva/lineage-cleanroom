"""Demo 2 — oversampling-before-split leakage on REAL data (Pima Indians Diabetes).

A very common real antipattern: duplicate/oversample rows to "balance" the classes, THEN split.
No fabricated features, no injected noise — just row duplication (what oversampling literally does).
Exact duplicates land on both sides, so Lineage CleanRoom's feature-hash gate catches them cleanly.

Run:  python -m demos.demo_oversampling_leak
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lineage_cleanroom import SplitView, audit_split, make_console_sink, write_report
from demos._common import auc_of, fetch_diabetes, OUT


def main():
    data = fetch_diabetes()
    if data is None:
        return None
    X, y, names = data
    rng = np.random.default_rng(0)

    # ANTIPATTERN: oversample (duplicate) 60% of rows to "balance", THEN split randomly.
    dup = rng.choice(len(y), size=int(0.6 * len(y)), replace=True)
    Xo = np.vstack([X, X[dup]])
    yo = np.concatenate([y, y[dup]])
    tr, te = train_test_split(np.arange(len(yo)), test_size=0.3, random_state=0, stratify=yo)

    view = SplitView(
        X_train=Xo[tr], y_train=yo[tr], X_test=Xo[te], y_test=yo[te], feature_names=names,
        prov_train=np.array(["human"] * len(tr), dtype=object),
        prov_test=np.array(["human"] * len(te), dtype=object))
    res = audit_split(view, emit=make_console_sink())

    auc_leaky = auc_of(Xo[tr], yo[tr], Xo[te], yo[te])
    # correct order: split the ORIGINAL rows first (no duplication across the boundary).
    otr, ote = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)
    auc_clean = auc_of(X[otr], y[otr], X[ote], y[ote])

    paths = write_report(res, OUT, stem="diabetes_oversampling_leak")
    (OUT / "diabetes_oversampling_leak.summary.txt").write_text(
        f"Dataset: Pima Indians Diabetes (OpenML, real) — oversample-before-split demo\n"
        f"AUC with leakage (oversample->split): {auc_leaky:.4f}\n"
        f"AUC honest (split first)            : {auc_clean:.4f}\n"
        f"exact feature-duplicate test rows   : {res.leak_report['feature_dup_test_rows']}"
        f" ({res.leak_report['test_leak_fraction']*100:.1f}% of test)\n"
        f"verdict                             : {res.verdict}\n", encoding="utf-8")

    print(f"\nDEMO oversampling: leaky AUC={auc_leaky:.4f}  honest AUC={auc_clean:.4f}")
    print(f"exact duplicates leaked: {res.leak_report['feature_dup_test_rows']} rows")
    print(f"verdict: {res.verdict}")
    return {"auc_leaky": auc_leaky, "auc_clean": auc_clean, "result": res}


if __name__ == "__main__":
    main()
