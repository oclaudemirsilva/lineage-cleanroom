# Lineage CleanRoom — audit report

**Status:** ❌ CONTAMINATION
**Verdict:** CONTAMINATION DETECTED - PROVENANCE - NON-HUMAN GOLD: 51 evaluation label(s) are not human-attested (distribution: {'human': 120, 'model': 51})
**Signed manifest verifies:** True
**Split:** 398 train / 171 test rows

## Leakage gate (features)
- test rows with an exact feature duplicate in train: **0**
- test leak fraction: **0.0**
- groups spanning train/test: **None**

## Provenance gate (label origin)
- human fraction in eval set: **0.7018**
- label sources (test): {'human': 120, 'model': 51}
- violations:
  - NON-HUMAN GOLD: 51 evaluation label(s) are not human-attested (distribution: {'human': 120, 'model': 51})

## Provenance manifest (Ed25519, tamper-evident)
The `.manifest.json` next to this file is cryptographically signed. Any change to a single
byte of its content invalidates the signature — attach it to a paper or submission as a
reproducibility certificate of this dataset's lineage and integrity verdict.
