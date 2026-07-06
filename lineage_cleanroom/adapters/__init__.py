"""lineage_cleanroom.adapters — adapters de formato (fonte -> SplitView).

Cada adapter carrega um formato concreto e devolve o mesmo contrato agnóstico ao domínio que o
núcleo consome (`SplitView`, definido em `lineage_cleanroom.ingest`). O CSV já é servido por
`ingest.load_split_from_single/_pair`; aqui ficam os formatos que dependem de bibliotecas
OPCIONAIS (import preguiçoso), para manter o núcleo leve.

Disponível: AnnData `.h5ad` (single-cell) via `load_split_from_anndata` — requer `anndata`.
"""
from __future__ import annotations

from .anndata_adapter import load_split_from_anndata

__all__ = ["load_split_from_anndata"]
