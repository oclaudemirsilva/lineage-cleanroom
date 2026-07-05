"""Lineage CleanRoom — provenance & contamination firewall for scientific ML.

Camada fina que detecta data leakage (vazamento treino/teste), audita a proveniência do rótulo
(gold = só mão humana; anti-autofagia), e emite um manifesto de proveniência assinado (Ed25519),
tamper-evident. Módulos com responsabilidade única e dependências injetadas (SOLID/DIP);
observabilidade via sink de telemetria.
"""
from .datagen import Dataset, make_leaky_dna_dataset
from .ingest import SplitView, load_split_from_pair, load_split_from_single
from .leakage import detect_leakage, sequences_to_kmer_features
from .manifest import build_manifest, dataset_fingerprint, sign_manifest, verify_manifest
from .pipeline import AuditResult, audit_split, run_catch_the_leak
from .provenance import ProvenancePolicy, audit_label_provenance, is_human, normalize_source
from .report import write_report
from .splits import make_group_aware_split, make_naive_random_split
from .telemetry import fanout, make_console_sink, make_jsonl_sink, null_sink

__all__ = [
    "Dataset", "make_leaky_dna_dataset",
    "SplitView", "load_split_from_single", "load_split_from_pair",
    "detect_leakage", "sequences_to_kmer_features",
    "ProvenancePolicy", "audit_label_provenance", "is_human", "normalize_source",
    "build_manifest", "dataset_fingerprint", "sign_manifest", "verify_manifest",
    "AuditResult", "audit_split", "run_catch_the_leak",
    "write_report",
    "make_group_aware_split", "make_naive_random_split",
    "fanout", "make_console_sink", "make_jsonl_sink", "null_sink",
]
