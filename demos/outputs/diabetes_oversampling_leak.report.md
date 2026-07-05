# Lineage CleanRoom — audit report

**Status:** ❌ CONTAMINATION
**Verdict:** CONTAMINATION DETECTED - LEAKAGE - 52.3% of test rows are exact feature-duplicates of training rows
**Signed manifest verifies:** True
**Split:** 859 train / 369 test rows

## Leakage gate (features)
- test rows with an exact feature duplicate in train: **193**
- test leak fraction: **0.523**
- groups spanning train/test: **None**

## Provenance gate (label origin)
- human fraction in eval set: **1.0**
- label sources (test): {'human': 369}
- no provenance violations

## Provenance manifest (Ed25519, tamper-evident)
The `.manifest.json` next to this file is cryptographically signed. Any change to a single
byte of its content invalidates the signature — attach it to a paper or submission as a
reproducibility certificate of this dataset's lineage and integrity verdict.
