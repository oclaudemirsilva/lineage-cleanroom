"""CI-verified demos. Offline tests always run; the OpenML (diabetes) tests skip if no network."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lineage_cleanroom import SplitView, audit_split
from demos._common import fetch_diabetes, replicate_as_patients


def test_provenance_gate_flags_model_generated_gold():
    d = load_breast_cancer()
    X, y, names = d.data, d.target.astype(int), list(d.feature_names)
    rng = np.random.default_rng(0)
    p = rng.permutation(len(y)); cut = int(0.7 * len(y)); tr, te = p[:cut], p[cut:]
    prov_test = np.array(["human"] * len(te), dtype=object)
    prov_test[rng.choice(len(te), int(0.3 * len(te)), replace=False)] = "model:imputed"
    view = SplitView(X[tr], y[tr], X[te], y[te], names,
                     prov_train=np.array(["human"] * len(tr), dtype=object), prov_test=prov_test)
    res = audit_split(view)
    assert res.leak_report["leaked"] is False          # no feature leakage
    assert res.passed is False                          # but provenance gate fails
    assert any("GOLD" in v for v in res.provenance_report["violations"])


def test_group_leak_mechanism_offline():
    # breast cancer replicated into "patients" -> a random split leaks the group; audit flags it.
    d = load_breast_cancer()
    Xs, ys, g = replicate_as_patients(d.data, d.target.astype(int), reps=3)
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(ys)), test_size=0.3, random_state=0, stratify=ys)
    view = SplitView(Xs[tr], ys[tr], Xs[te], ys[te], list(d.feature_names),
                     groups_train=g[tr], groups_test=g[te])
    res = audit_split(view)
    assert res.leak_report["group_span_test_rows"] > 0
    assert res.passed is False


@pytest.mark.parametrize("mod", ["demo_diabetes_group_leak", "demo_oversampling_leak"])
def test_diabetes_demos_if_online(mod):
    if fetch_diabetes() is None:
        pytest.skip("OpenML unavailable (offline)")
    import importlib
    m = importlib.import_module(f"demos.{mod}")
    out = m.main()
    assert out is not None
    assert out["result"].passed is False                # contamination detected on real data
    assert out["auc_leaky"] > out["auc_clean"]          # leakage inflated the metric
