---
name: lineage-cleanroom
description: Audit a scientific ML dataset split for data leakage and label-provenance contamination, and emit a cryptographically signed reproducibility manifest. Use when a researcher asks "is my train/test split leaking?", "can I trust this AUC/accuracy?", "are my gold labels human-attested?", or before publishing/submitting a model or figure. Works on any tabular CSV (genomics, single-cell/CRISPR screens, protein-interaction features, clinical tables).
---

# Lineage CleanRoom

A provenance & contamination firewall for scientific ML. It plugs into a Claude Science / Claude Code
workflow and answers the question a reviewer, clinician, or regulator actually asks: **can I trust this
number?** — then signs the answer so it cannot be quietly altered.

## When to use
- A model's accuracy/AUC looks suspiciously high and you want to check for train/test leakage.
- You need to certify that a dataset's evaluation labels are human-attested (not model-generated).
- You want a tamper-evident provenance certificate to attach to a paper, submission, or model card.

## Two gates
1. **Leakage gate (features):** flags test rows that are exact feature-duplicates of training rows, and
   groups (donor / batch / genomic region) that span the train/test boundary — the silent cause of
   inflated metrics.
2. **Provenance gate (label origin):** classifies each label's source (human / model / heuristic /
   unknown), enforces that gold/evaluation labels are human-attested only, and can forbid
   model-generated labels in training (anti-autophagy).

Output: a **signed Ed25519 manifest** (`*.manifest.json`) + a human-readable report (`*.report.md`).

## How to run

Audit a single CSV that has a split column:
```
python -m lineage_cleanroom scan \
  --data dataset.csv --label activity --split-col split \
  --group donor --provenance label_source --out ./audit
```

Audit a pair of files (train.csv / test.csv):
```
python -m lineage_cleanroom scan \
  --train train.csv --test test.csv --label activity --group donor --out ./audit
```

Flags: `--features a,b,c` (default: all columns except declared roles) · `--forbid-model-train`
(anti-autophagy) · `--allow-nonhuman-gold` (relax the gold-human rule) · `--telemetry events.jsonl`.

**Exit code:** `0` = clean · `1` = contamination detected (CI-friendly) · `2` = usage error.

Demonstration on synthetic data (the "catch-the-leak" before/after):
```
python -m lineage_cleanroom demo
```

## Column roles
| flag | meaning |
|---|---|
| `--label` | the target column (y) |
| `--features` | feature columns (optional; inferred otherwise) |
| `--group` | grouping id: donor, batch, genomic region — used to detect group-spanning leakage |
| `--provenance` | label origin per row: `human`, `model:<name>`, `heuristic`, `unknown` |
| `--split-col` / `--train`+`--test` | how the train/test split is defined |

## Notes
- Deterministic and offline; no network, no model training required for the audit.
- The synthetic generator (`datagen.py`) is a test fixture, not the product — point the tool at real data.
