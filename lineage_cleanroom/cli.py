"""lineage_cleanroom/cli.py — interface de linha de comando (Development Track).

Uso:
  # audita um split real (um arquivo com coluna de split, ou dois arquivos):
  python -m lineage_cleanroom scan --data data.csv --label y --split-col split \
      --group donor --provenance label_source --out ./out
  python -m lineage_cleanroom scan --train train.csv --test test.csv --label y --group donor

  # demonstração (catch-the-leak em dado sintético):
  python -m lineage_cleanroom demo

Exit-code (amigável a CI): 0 = limpo; 1 = contaminação detectada; 2 = erro de uso.
"""
from __future__ import annotations

import argparse
import sys

from .adapters import load_split_from_anndata
from .ingest import load_split_from_pair, load_split_from_single
from .pipeline import audit_split
from .provenance import ProvenancePolicy
from .report import write_report
from .telemetry import fanout, make_console_sink, make_jsonl_sink


def _split_list(s: str | None) -> list[str] | None:
    return [c.strip() for c in s.split(",") if c.strip()] if s else None


def cmd_scan(args: argparse.Namespace) -> int:
    if bool(args.data) == bool(args.train or args.test):
        print("erro: forneça OU --data (com --split-col) OU --train e --test", file=sys.stderr)
        return 2

    features = _split_list(args.features)
    if args.data:
        if args.data.lower().endswith((".h5ad", ".h5")):
            # single-cell: mesmo contrato de papéis, container AnnData (dep opcional, import lazy).
            view = load_split_from_anndata(
                args.data, label_col=args.label, split_col=args.split_col,
                feature_source=args.feature_source, obsm_key=args.obsm_key,
                group_col=args.group, provenance_col=args.provenance,
                train_value=args.train_value, test_value=args.test_value)
        else:
            view = load_split_from_single(
                args.data, label_col=args.label, split_col=args.split_col,
                feature_cols=features, group_col=args.group, provenance_col=args.provenance,
                train_value=args.train_value, test_value=args.test_value)
    else:
        if not (args.train and args.test):
            print("erro: --train e --test são ambos obrigatórios nesse modo", file=sys.stderr)
            return 2
        view = load_split_from_pair(
            args.train, args.test, label_col=args.label,
            feature_cols=features, group_col=args.group, provenance_col=args.provenance)

    sinks = [make_console_sink()]
    if args.telemetry:
        sinks.append(make_jsonl_sink(args.telemetry))
    emit = fanout(*sinks)

    policy = ProvenancePolicy(
        require_human_gold=not args.allow_nonhuman_gold,
        forbid_model_labels_in_train=args.forbid_model_train)

    result = audit_split(view, policy=policy, emit=emit)

    print("\n" + "=" * 64)
    print(f"  Lineage CleanRoom -- {'CLEAN [PASS]' if result.passed else 'CONTAMINATION [FAIL]'}")
    print("=" * 64)
    print(f"  {result.verdict}")
    print(f"  signed manifest verifies: {result.verify_ok}")
    if args.out:
        paths = write_report(result, args.out)
        print(f"  manifest -> {paths['manifest']}")
        print(f"  report   -> {paths['report']}")
    print("=" * 64)
    return 0 if result.passed else 1


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo_catch_the_leak import main as demo_main
    out = demo_main(jsonl_path=args.telemetry)
    ok = out["leak_report"]["leaked"] and out["verify_ok"] and not out["verify_after_tamper"]
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lineage_cleanroom",
                                description="Provenance & contamination firewall for scientific ML.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="audita um split treino/teste real")
    s.add_argument("--data", help="CSV único, ou AnnData .h5ad (single-cell); requer --split-col")
    s.add_argument("--feature-source", default="X",
                   help="para .h5ad: 'X' (matriz células×genes) ou 'obsm' (incorporação nomeada)")
    s.add_argument("--obsm-key", help="para .h5ad com --feature-source obsm: chave em .obsm (ex.: X_pca)")
    s.add_argument("--train", help="CSV de treino (modo par)")
    s.add_argument("--test", help="CSV de teste (modo par)")
    s.add_argument("--label", required=True, help="coluna do rótulo (y)")
    s.add_argument("--features", help="colunas de feature, separadas por vírgula (default: todas as demais)")
    s.add_argument("--group", help="coluna de grupo (doador/batch/região)")
    s.add_argument("--provenance", help="coluna de proveniência do rótulo (ex.: human/model:...)")
    s.add_argument("--split-col", default="split", help="coluna de split no modo --data (default: split)")
    s.add_argument("--train-value", default="train")
    s.add_argument("--test-value", default="test")
    s.add_argument("--allow-nonhuman-gold", action="store_true",
                   help="NÃO exigir rótulo humano no conjunto de avaliação")
    s.add_argument("--forbid-model-train", action="store_true",
                   help="reprovar rótulos gerados por modelo no treino (anti-autofagia)")
    s.add_argument("--out", help="diretório para escrever manifest.json + report.md")
    s.add_argument("--telemetry", help="caminho JSONL para eventos de observabilidade")
    s.set_defaults(func=cmd_scan)

    d = sub.add_parser("demo", help="catch-the-leak em dado sintético (demonstração)")
    d.add_argument("--telemetry", help="caminho JSONL para eventos")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
