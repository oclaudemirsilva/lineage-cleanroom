"""Testes do adapter AnnData (.h5ad) — com CONTROLE POSITIVO e CONTROLE NEGATIVO.

Rigor do próprio projeto: não basta provar que acusa o dado SUJO (split que vaza doador); é
preciso provar que NÃO acusa o dado LIMPO (split group-aware). Se `anndata` não estiver instalado,
os testes são pulados (dependência opcional — o núcleo não a exige).
"""
from __future__ import annotations

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")

from lineage_cleanroom.adapters import load_split_from_anndata
from lineage_cleanroom.pipeline import audit_split
from lineage_cleanroom.provenance import ProvenancePolicy

N_DONORS = 12
CELLS_PER_DONOR = 10
N_GENES = 20


def _make_adata(sparse: bool = False):
    """Constrói um AnnData sintético com dois splits: `split_leak` (todo doador cruza treino/teste)
    e `split_clean` (doadores inteiros de um lado só). Rótulos numéricos; proveniência 'human'."""
    import pandas as pd
    from scipy import sparse as sp

    rng = np.random.default_rng(0)
    n = N_DONORS * CELLS_PER_DONOR
    X = rng.normal(size=(n, N_GENES)).astype(np.float32)

    donor = np.repeat([f"d{i}" for i in range(N_DONORS)], CELLS_PER_DONOR)
    label = rng.integers(0, 2, size=n)

    # split_leak: dentro de CADA doador, 6 treino / 4 teste -> todo doador cruza a fronteira.
    split_leak = np.empty(n, dtype=object)
    for i in range(N_DONORS):
        s = i * CELLS_PER_DONOR
        split_leak[s:s + 6] = "train"
        split_leak[s + 6:s + CELLS_PER_DONOR] = "test"

    # split_clean: doadores d0..d7 -> treino; d8..d11 -> teste (group-aware, nenhum cruza).
    split_clean = np.where(np.isin(donor, [f"d{i}" for i in range(8)]), "train", "test").astype(object)

    prov = np.full(n, "human", dtype=object)

    obs = pd.DataFrame({
        "donor": donor, "label": label,
        "split_leak": split_leak, "split_clean": split_clean, "prov": prov,
    })
    Xmat = sp.csr_matrix(X) if sparse else X
    ad = anndata.AnnData(X=Xmat, obs=obs)
    ad.var_names = [f"gene{i}" for i in range(N_GENES)]
    ad.obsm["X_emb"] = rng.normal(size=(n, 5)).astype(np.float32)
    return ad


def test_positive_control_donor_leak_is_caught(tmp_path):
    p = tmp_path / "leak.h5ad"
    _make_adata().write_h5ad(p)

    view = load_split_from_anndata(
        str(p), label_col="label", split_col="split_leak", group_col="donor")
    res = audit_split(view)

    assert res.leak_report["group_span_test_rows"] > 0
    assert res.passed is False
    assert "LEAKAGE" in res.verdict
    assert res.verify_ok is True


def test_negative_control_clean_split_not_flagged(tmp_path):
    """Controle negativo: split group-aware não deve disparar vazamento (nada de crying wolf)."""
    p = tmp_path / "clean.h5ad"
    _make_adata().write_h5ad(p)

    view = load_split_from_anndata(
        str(p), label_col="label", split_col="split_clean",
        group_col="donor", provenance_col="prov")
    res = audit_split(view)

    assert res.leak_report["group_span_test_rows"] == 0
    assert res.leak_report["feature_dup_test_rows"] == 0
    assert res.passed is True
    assert "CLEAN" in res.verdict


def test_provenance_gate_flags_model_gold(tmp_path):
    """Rótulos de modelo no conjunto de avaliação -> gate de proveniência reprova (gold só humano)."""
    ad = _make_adata()
    prov = ad.obs["prov"].to_numpy().copy()
    is_test = ad.obs["split_clean"].to_numpy() == "test"
    idx = np.flatnonzero(is_test)[:5]
    prov[idx] = "model:foo"
    ad.obs["prov"] = prov
    p = tmp_path / "prov.h5ad"
    ad.write_h5ad(p)

    view = load_split_from_anndata(
        str(p), label_col="label", split_col="split_clean", provenance_col="prov")
    res = audit_split(view, policy=ProvenancePolicy(require_human_gold=True))

    assert res.passed is False
    assert res.provenance_report["provided"] is True
    assert any("NON-HUMAN GOLD" in v for v in res.provenance_report["violations"])


def test_obsm_feature_source(tmp_path):
    p = tmp_path / "emb.h5ad"
    _make_adata().write_h5ad(p)

    view = load_split_from_anndata(
        str(p), label_col="label", split_col="split_clean",
        feature_source="obsm", obsm_key="X_emb")

    assert view.X_train.shape[1] == 5
    assert len(view.feature_names) == 5


def test_sparse_X_is_densified(tmp_path):
    p = tmp_path / "sparse.h5ad"
    _make_adata(sparse=True).write_h5ad(p)

    view = load_split_from_anndata(
        str(p), label_col="label", split_col="split_leak", group_col="donor")

    assert isinstance(view.X_train, np.ndarray)
    assert view.X_train.shape[1] == N_GENES


def test_missing_column_raises(tmp_path):
    p = tmp_path / "m.h5ad"
    _make_adata().write_h5ad(p)
    with pytest.raises(ValueError, match="não encontradas"):
        load_split_from_anndata(str(p), label_col="nope", split_col="split_leak")


def test_obsm_source_without_key_raises(tmp_path):
    p = tmp_path / "ok.h5ad"
    _make_adata().write_h5ad(p)
    with pytest.raises(ValueError, match="obsm_key"):
        load_split_from_anndata(
            str(p), label_col="label", split_col="split_clean", feature_source="obsm")


def test_invalid_feature_source_raises(tmp_path):
    p = tmp_path / "ok.h5ad"
    _make_adata().write_h5ad(p)
    with pytest.raises(ValueError, match="feature_source"):
        load_split_from_anndata(
            str(p), label_col="label", split_col="split_clean", feature_source="bogus")


def test_split_value_absent_raises(tmp_path):
    p = tmp_path / "ok.h5ad"
    _make_adata().write_h5ad(p)
    with pytest.raises(ValueError, match="Nenhum exemplo"):
        load_split_from_anndata(
            str(p), label_col="label", split_col="split_clean", test_value="holdout")
