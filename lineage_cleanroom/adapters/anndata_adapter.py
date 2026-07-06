"""lineage_cleanroom/adapters/anndata_adapter.py — adapter de formato: AnnData (.h5ad) -> SplitView.

AnnData é o container padrão de single-cell (scanpy / Perturb-seq / telas CRISPR). Este adapter
NÃO embute biologia: mapeia os PAPÉIS das colunas — quais colunas de `.obs` são rótulo / grupo /
proveniência / split são ARGUMENTOS do chamador, exatamente como no adapter de CSV (`ingest.py`).

  - `.X`         → matriz células × genes (features); ou uma incorporação nomeada em `.obsm`.
  - `.obs`       → metadados por célula (rótulo, grupo=doador/batch, proveniência, split).
  - `.var_names` → nomes das features (genes).

`anndata` é dependência OPCIONAL (import preguiçoso): o núcleo continua leve (numpy/sklearn/
cryptography). Determinístico, offline, SRP (só carrega e valida — nada de detecção de vazamento).

Corpo gerado pelo agente `code` (DeepSeek via Super Squad) sob especificação e gate humano/Claude.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from ..ingest import SplitView


def _require_obs_cols(obs, cols: list[Optional[str]]) -> None:
    """
    Valida que as colunas solicitadas existem no DataFrame .obs.
    Ignora valores None na lista de colunas.
    Levanta ValueError com as colunas disponíveis se alguma não for encontrada.
    """
    required = [c for c in cols if c is not None]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        available = list(obs.columns)
        raise ValueError(
            f"Colunas não encontradas em .obs: {missing}. "
            f"Colunas disponíveis: {available}"
        )


def load_split_from_anndata(
    path: str,
    *,
    label_col: str,
    split_col: str,
    feature_source: str = "X",
    obsm_key: Optional[str] = None,
    group_col: Optional[str] = None,
    provenance_col: Optional[str] = None,
    train_value: str = "train",
    test_value: str = "test",
) -> SplitView:
    """
    Carrega um arquivo AnnData (.h5ad) e produz uma SplitView mapeando colunas de .obs.

    AnnData é o container padrão para dados single-cell:
    - .X: matriz células × genes (features)
    - .obs: metadados por célula (rótulo, grupo, proveniência, split)
    - .var_names: nomes das features (genes)

    Sem biologia embutida: os papéis das colunas são definidos pelos argumentos.

    Args:
        path: Caminho para o arquivo .h5ad
        label_col: Coluna em .obs contendo os rótulos alvo
        split_col: Coluna em .obs definindo split train/test
        feature_source: Origem das features ("X" ou "obsm")
        obsm_key: Chave em .obsm quando feature_source="obsm"
        group_col: Coluna em .obs para grupos (ex: paciente)
        provenance_col: Coluna em .obs para proveniência (ex: lote)
        train_value: Valor em split_col que indica treino
        test_value: Valor em split_col que indica teste

    Returns:
        SplitView com dados separados por split

    Raises:
        ImportError: Se anndata não estiver instalado
        ValueError: Para argumentos inválidos ou dados inconsistentes
    """
    try:
        import anndata
    except ImportError as e:
        raise ImportError(
            "Adapter .h5ad requer 'anndata' (pip install anndata). "
            "AnnData é uma dependência opcional do Lineage CleanRoom."
        ) from e

    # Carrega o arquivo AnnData
    adata = anndata.read_h5ad(path)

    # Valida colunas obrigatórias e opcionais
    _require_obs_cols(adata.obs, [label_col, split_col, group_col, provenance_col])

    # Extrai matriz de features
    if feature_source == "X":
        Xmat = adata.X
        feature_names = [str(v) for v in adata.var_names]
    elif feature_source == "obsm":
        if obsm_key is None:
            raise ValueError(
                "obsm_key é obrigatório quando feature_source='obsm'"
            )
        if obsm_key not in adata.obsm:
            available = list(adata.obsm.keys())
            raise ValueError(
                f"Chave '{obsm_key}' não encontrada em .obsm. "
                f"Chaves disponíveis: {available}"
            )
        Xmat = adata.obsm[obsm_key]
        feature_names = [f"{obsm_key}[{i}]" for i in range(Xmat.shape[1])]
    else:
        raise ValueError(
            f"feature_source deve ser 'X' ou 'obsm', recebido: {feature_source}"
        )

    # Converte para array denso se necessário
    try:
        import scipy.sparse
        if scipy.sparse.issparse(Xmat):
            Xmat = Xmat.toarray()
    except ImportError:
        # Se scipy não estiver disponível, assume que não é esparso
        pass

    # Garante que Xmat é um array NumPy de floats
    Xmat = np.asarray(Xmat, dtype=float)

    # Extrai split e cria máscaras
    split = adata.obs[split_col].to_numpy()
    is_train = split == train_value
    is_test = split == test_value

    # Valida que há exemplos em ambos os splits
    if is_train.sum() == 0:
        found = sorted(set(split))
        raise ValueError(
            f"Nenhum exemplo com split='{train_value}'. "
            f"Valores encontrados: {found}"
        )
    if is_test.sum() == 0:
        found = sorted(set(split))
        raise ValueError(
            f"Nenhum exemplo com split='{test_value}'. "
            f"Valores encontrados: {found}"
        )

    # Extrai rótulos
    y = adata.obs[label_col].to_numpy()
    y_train = y[is_train]
    y_test = y[is_test]

    # Extrai grupos se fornecido
    if group_col is not None:
        groups = adata.obs[group_col].to_numpy()
        groups_train = groups[is_train]
        groups_test = groups[is_test]
    else:
        groups_train = groups_test = None

    # Extrai proveniência se fornecido
    if provenance_col is not None:
        prov = adata.obs[provenance_col].to_numpy()
        prov_train = prov[is_train]
        prov_test = prov[is_test]
    else:
        prov_train = prov_test = None

    # Cria e retorna a SplitView
    return SplitView(
        X_train=Xmat[is_train],
        y_train=y_train,
        X_test=Xmat[is_test],
        y_test=y_test,
        feature_names=feature_names,
        groups_train=groups_train,
        groups_test=groups_test,
        prov_train=prov_train,
        prov_test=prov_test,
    )
