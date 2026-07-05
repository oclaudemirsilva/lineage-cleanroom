"""lineage_cleanroom/demo_catch_the_leak.py — demo E2E rodável (o "catch-the-leak").

Fia as dependências REAIS (dataset sintético, RandomForest, splits, telemetria) no core
`run_catch_the_leak` e imprime o antes/depois. Rode:  python -m lineage_cleanroom demo
"""
from __future__ import annotations

import json
import sys

from sklearn.ensemble import RandomForestClassifier

from .datagen import make_leaky_dna_dataset
from .pipeline import run_catch_the_leak
from .splits import make_group_aware_split, make_naive_random_split
from .telemetry import fanout, make_console_sink, make_jsonl_sink


def main(jsonl_path: str | None = None) -> dict:
    dataset = make_leaky_dna_dataset(seed=7)

    sinks = [make_console_sink()]
    if jsonl_path:
        sinks.append(make_jsonl_sink(jsonl_path))
    emit = fanout(*sinks)

    result = run_catch_the_leak(
        dataset,
        model_factory=lambda: RandomForestClassifier(n_estimators=120, random_state=0, n_jobs=-1),
        naive_split_fn=make_naive_random_split(test_size=0.3, seed=0),
        clean_split_fn=make_group_aware_split(test_size=0.3, seed=0),
        emit=emit,
    )

    print("\n" + "=" * 66)
    print("  Lineage CleanRoom — catch-the-leak (DNA regulatory-activity, sintético)")
    print("=" * 66)
    print(f"  AUC (split ingênuo, VAZADO) ........ {result['auc_naive']:.4f}   <- inflada")
    print(f"  Vazamento detectado ................ "
          f"{result['leak_report']['test_leak_fraction']*100:.1f}% das linhas de teste"
          f" ({result['leak_report']['feature_dup_test_rows']} duplicatas de features,"
          f" {result['leak_report']['group_span_test_rows']} grupos cruzando)")
    print(f"  AUC (split LIMPO, group-aware) ..... {result['auc_clean']:.4f}   <- honesta")
    print(f"  Inflação por vazamento ............. {result['inflation']:.4f}")
    print(f"  Vazamento residual no split limpo .. "
          f"{result['leak_report_clean']['test_leak_fraction']*100:.1f}%")
    print("-" * 66)
    print(f"  Veredito: {result['verdict']}")
    print(f"  Manifesto assinado verifica ........ {result['verify_ok']}")
    print(f"  Verifica APÓS adulterar 1 campo .... {result['verify_after_tamper']}   <- rejeita a fraude")
    print("=" * 66)
    print("\n  Manifesto assinado (Ed25519):")
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    out = main(jsonl_path=sys.argv[1] if len(sys.argv) > 1 else None)
    # exit-code acionável: 0 se o firewall funcionou (pegou o vazamento E rejeitou a adulteração).
    ok = out["leak_report"]["leaked"] and out["verify_ok"] and not out["verify_after_tamper"]
    sys.exit(0 if ok else 1)
