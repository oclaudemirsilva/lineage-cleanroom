"""Demo 1 (flagship) — patient-level leakage on REAL data (Pima Indians Diabetes, OpenML).

The #1 real medical-ML mistake: multiple measurements per patient, then a random row-level split
puts the same patient in train AND test. A model memorizes the patient instead of learning to
generalize. Lineage CleanRoom flags the group-spanning leakage; a group-aware split reveals the
honest performance.

Run:  python -m demos.demo_diabetes_group_leak
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root importable
from lineage_cleanroom import SplitView, audit_split, make_console_sink, write_report
from demos._common import auc_of, fetch_diabetes, replicate_as_patients, OUT


def main():
    data = fetch_diabetes()
    if data is None:
        return None
    X, y, names = data
    Xs, ys, groups = replicate_as_patients(X, y, reps=3)

    # LEAKY: random split by row -> the same patient lands in train and test.
    tr, te = train_test_split(np.arange(len(ys)), test_size=0.3, random_state=0, stratify=ys)
    view = SplitView(
        X_train=Xs[tr], y_train=ys[tr], X_test=Xs[te], y_test=ys[te], feature_names=names,
        groups_train=groups[tr], groups_test=groups[te],
        prov_train=np.array(["human"] * len(tr), dtype=object),
        prov_test=np.array(["human"] * len(te), dtype=object))
    res = audit_split(view, emit=make_console_sink())

    # what the leak does to the metric, vs an honest group-aware split
    auc_leaky = auc_of(Xs[tr], ys[tr], Xs[te], ys[te])
    ctr, cte = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(Xs, ys, groups))
    auc_clean = auc_of(Xs[ctr], ys[ctr], Xs[cte], ys[cte])

    paths = write_report(res, OUT, stem="diabetes_group_leak")
    (OUT / "diabetes_group_leak.summary.txt").write_text(
        f"Dataset: Pima Indians Diabetes (OpenML, real) — patient-level leakage demo\n"
        f"AUC with leakage (random row split) : {auc_leaky:.4f}\n"
        f"AUC honest (group-aware by patient) : {auc_clean:.4f}\n"
        f"inflation from leakage              : {auc_leaky - auc_clean:+.4f}\n"
        f"verdict                             : {res.verdict}\n"
        f"signed manifest verifies            : {res.verify_ok}\n", encoding="utf-8")

    print(f"\nDEMO diabetes group-leak: leaky AUC={auc_leaky:.4f}  honest AUC={auc_clean:.4f}"
          f"  (inflation {auc_leaky - auc_clean:+.4f})")
    print(f"verdict: {res.verdict}")
    print(f"outputs -> {paths['manifest']}")
    return {"auc_leaky": auc_leaky, "auc_clean": auc_clean, "result": res}


if __name__ == "__main__":
    main()
