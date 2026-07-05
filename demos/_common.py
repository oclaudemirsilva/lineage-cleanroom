"""Shared helpers for the demos (real public datasets)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent / "outputs"


def rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)


def auc_of(X_tr, y_tr, X_te, y_te) -> float:
    m = rf()
    m.fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))


def fetch_diabetes():
    """Real Pima Indians Diabetes (OpenML). Returns (X, y, feature_names) or None if offline."""
    try:
        import socket
        from sklearn.datasets import fetch_openml
        socket.setdefaulttimeout(20)
        d = fetch_openml(name="diabetes", version=1, as_frame=True)
        X = d.data.to_numpy(float)
        y = (d.target.to_numpy() == "tested_positive").astype(int)
        return X, y, list(d.data.columns)
    except Exception as e:  # offline / OpenML down — evidence is already committed in demos/outputs/
        print(f"[demo] OpenML unavailable ({type(e).__name__}) — skipping live run; "
              f"see committed evidence in demos/outputs/.")
        return None


def replicate_as_patients(X, y, reps: int = 3, noise: float = 0.01, seed: int = 0):
    """Model repeated per-patient measurements: each row becomes a 'patient' with `reps` noisy
    replicates sharing the same label. The FEATURES and labels are real; the grouping models a
    real clinical scenario (multiple readings per patient) so we can demonstrate patient leakage."""
    rng = np.random.default_rng(seed)
    Xs, ys, groups = [], [], []
    for i in range(len(y)):
        for _ in range(reps):
            Xs.append(X[i] + rng.normal(0, noise * (np.abs(X[i]) + 1)))
            ys.append(int(y[i]))
            groups.append(i)
    return np.array(Xs), np.array(ys), np.array(groups)
