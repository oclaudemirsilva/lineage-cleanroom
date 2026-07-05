"""lineage_cleanroom/ingest.py — ingestão de dataset REAL de arquivo (SRP: só carrega e valida).

O produto opera sobre dados de verdade, não sintéticos. Aqui carregamos um CSV (ou par de CSVs
treino/teste) declarando o PAPEL de cada coluna: features, rótulo, grupo (doador/batch/região),
proveniência do rótulo, e o split. Nada de biologia embutida — o adapter é agnóstico ao domínio.

Duas formas de definir o split (o caso real do pesquisador: "eu já separei — está vazando?"):
  - `load_split_from_single`: um arquivo com uma COLUNA de split (train/test).
  - `load_split_from_pair`: dois arquivos (train.csv, test.csv).

Devolve um `SplitView` — a estrutura que o pipeline de auditoria consome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitView:
    """Uma visão treino/teste já materializada em arrays (baixo acoplamento: o pipeline não
    conhece pandas nem arquivos, só este contrato)."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    groups_train: "Optional[np.ndarray]" = None
    groups_test: "Optional[np.ndarray]" = None
    prov_train: "Optional[np.ndarray]" = None
    prov_test: "Optional[np.ndarray]" = None

    @property
    def n_train(self) -> int:
        return int(len(self.y_train))

    @property
    def n_test(self) -> int:
        return int(len(self.y_test))


def _infer_feature_cols(
    df: pd.DataFrame, label_col: str, exclude: Sequence[Optional[str]]
) -> list[str]:
    drop = {label_col, *[c for c in exclude if c]}
    return [c for c in df.columns if c not in drop]


def _col(df: pd.DataFrame, name: Optional[str]) -> "Optional[np.ndarray]":
    return df[name].to_numpy() if name else None


def load_split_from_single(
    path: str,
    *,
    label_col: str,
    split_col: str,
    feature_cols: "Optional[Sequence[str]]" = None,
    group_col: "Optional[str]" = None,
    provenance_col: "Optional[str]" = None,
    train_value: str = "train",
    test_value: str = "test",
) -> SplitView:
    """Carrega UM CSV que já tem coluna de split. `feature_cols=None` => todas menos os papéis."""
    df = pd.read_csv(path)
    _require_cols(df, [label_col, split_col, group_col, provenance_col])
    feats = list(feature_cols) if feature_cols else _infer_feature_cols(
        df, label_col, [split_col, group_col, provenance_col])

    tr = df[df[split_col] == train_value]
    te = df[df[split_col] == test_value]
    if len(tr) == 0 or len(te) == 0:
        raise ValueError(
            f"split '{split_col}' não produziu treino E teste "
            f"(valores esperados: {train_value!r}/{test_value!r}; achados: {sorted(df[split_col].unique())})")

    return SplitView(
        X_train=tr[feats].to_numpy(dtype=float), y_train=tr[label_col].to_numpy(),
        X_test=te[feats].to_numpy(dtype=float), y_test=te[label_col].to_numpy(),
        feature_names=feats,
        groups_train=_col(tr, group_col), groups_test=_col(te, group_col),
        prov_train=_col(tr, provenance_col), prov_test=_col(te, provenance_col),
    )


def load_split_from_pair(
    train_path: str,
    test_path: str,
    *,
    label_col: str,
    feature_cols: "Optional[Sequence[str]]" = None,
    group_col: "Optional[str]" = None,
    provenance_col: "Optional[str]" = None,
) -> SplitView:
    """Carrega DOIS CSVs (treino, teste) — o caso em que o pesquisador já tem os arquivos separados."""
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    for df in (tr, te):
        _require_cols(df, [label_col, group_col, provenance_col])
    feats = list(feature_cols) if feature_cols else _infer_feature_cols(
        tr, label_col, [group_col, provenance_col])

    return SplitView(
        X_train=tr[feats].to_numpy(dtype=float), y_train=tr[label_col].to_numpy(),
        X_test=te[feats].to_numpy(dtype=float), y_test=te[label_col].to_numpy(),
        feature_names=feats,
        groups_train=_col(tr, group_col), groups_test=_col(te, group_col),
        prov_train=_col(tr, provenance_col), prov_test=_col(te, provenance_col),
    )


def _require_cols(df: pd.DataFrame, cols: Sequence[Optional[str]]) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ValueError(f"colunas ausentes no arquivo: {missing}. Colunas disponíveis: {list(df.columns)}")
