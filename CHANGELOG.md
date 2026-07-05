# Changelog

All notable changes to Lineage CleanRoom. Kept human-readable; the *why* lives in [DECISIONS.md](DECISIONS.md).

## [0.1.1] — 2026-07-05
### Added
- **Demonstrations on real public data** ([`demos/`](demos/)), each with a committed signed manifest and CI-verified:
  - Patient-level (group) leakage on **Pima Indians Diabetes** (OpenML): AUC 0.9865 → 0.8246 honest.
  - Oversample-before-split on Pima diabetes: 193 exact feature-duplicates caught.
  - Label-provenance gate on **Breast Cancer Wisconsin**: 51 model-generated evaluation labels flagged.
- **Known limitations & roadmap** section (README) — states honestly what the tool does *not* detect.

### Changed
- Audit verdict now reports **group-spanning** leakage explicitly (previously only the exact-duplicate
  fraction, which read as "0.0%" on group leaks — misleading).
- Provenance violation messages translated to English (they appear in the signed manifest that reviewers read).

### Investigated but not shipped
- **Near-duplicate leakage detection** (to catch augmentation/jitter that the exact-hash gate misses).
  Empirically, distance-based detection cannot separate moderate augmentation from legitimately-similar
  independent samples without false positives (measured on Pima diabetes). Removed rather than ship an
  overclaiming detector; documented as a limitation + roadmap. See [DECISIONS.md](DECISIONS.md) (DEC-5).

## [0.1.0] — 2026-07-04
### Added
- Two integrity gates: **leakage** (exact feature-hash + group-aware) and **label provenance**
  (human-attested gold; optional anti-autophagy).
- **Ed25519-signed, tamper-evident manifest** of the dataset lineage and verdict.
- CSV ingestion (`ingest`), CLI (`scan` / `demo`), injected telemetry, Claude Code skill (`SKILL.md`).
- Synthetic "catch-the-leak" demonstrator (AUC 0.9995 → 0.6532 honest; manifest rejects tampering).
