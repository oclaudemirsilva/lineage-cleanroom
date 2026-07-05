# Lineage CleanRoom — audit report

**Status:** ❌ CONTAMINATION
**Verdict:** CONTAMINATION DETECTED - LEAKAGE - 647 test rows share a group (donor/batch/region) with training
**Signed manifest verifies:** True
**Split:** 1612 train / 692 test rows

## Leakage gate (features)
- test rows with an exact feature duplicate in train: **0**
- test leak fraction: **0.0**
- groups spanning train/test: **647**

## Provenance gate (label origin)
- human fraction in eval set: **1.0**
- label sources (test): {'human': 692}
- no provenance violations

## Provenance manifest (Ed25519, tamper-evident)
The `.manifest.json` next to this file is cryptographically signed. Any change to a single
byte of its content invalidates the signature — attach it to a paper or submission as a
reproducibility certificate of this dataset's lineage and integrity verdict.
