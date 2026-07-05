# Decision records

Short records of the load-bearing decisions and *why* — including the ones we reversed. A provenance
tool should keep its own lineage of decisions. Newest first.

## DEC-5 — Near-duplicate detection: investigated, measured, not shipped
**Context.** The exact-hash gate misses *near*-duplicates (data augmentation / jitter / noisy resampling
across the split) — a real, common leakage source. We set out to close it.
**What we tried.** Three distance-based detectors in standardized feature space: (a) threshold vs the
train-internal nearest-neighbor spacing; (b) an absolute `beta·√d` threshold; (c) a null-calibrated
threshold (percentile of nearest-neighbor distances between two independent halves of the training set).
**Finding (measured on Pima diabetes + breast cancer).** (a) is self-defeating — augmentation shrinks the
train-internal reference, hiding itself. (b) flags 20–99% of *clean* data (false positives) in low
dimensions. (c) gives zero false positives but only catches near-*exact* copies; moderate augmentation
(jitter ≈ 0.005·|x|) lands at the same distances as legitimately-similar independent samples — no global
threshold separates them.
**Decision.** Do **not** ship a detector that overclaims. Removed the code; documented the gap as a known
limitation + roadmap. Robust detection needs **provenance tracking** (record augmentation lineage) or
learned embeddings — not raw distance. Shipping honesty over a flashy-but-shaky feature is the whole point
of this tool.

## DEC-4 — Demonstrate on Pima diabetes, not breast cancer
**Decision.** Use a *modest-baseline* real dataset (Pima diabetes, honest AUC ≈ 0.82) as the flagship demo.
**Why.** Breast Cancer Wisconsin is too easy (honest AUC ≈ 0.99) — there is no headroom for leakage to
inflate, so it can't *demonstrate* the problem. We kept breast cancer for the provenance demo (which
doesn't need headroom) and RandomForest for the classifier (KNN under-learns the signal in high dimensions,
making the honest baseline look like noise).

## DEC-3 — Name: "Lineage CleanRoom"
**Decision.** Compound name: *lineage* (data lineage + cell lineage) + *cleanroom* (contamination control).
**Why.** "Cleanroom" alone is legible but generic and names only the *state* (clean), missing the
provenance/attestation half. "Lineage" carries the essence (where data/labels came from) and has a
biology double meaning judges read instantly. Rejected: plain "CleanRoom" (generic), "Digital CleanRoom"
(collides with adtech "data clean rooms").

## DEC-2 — Build Track, positioned as a firewall *additive* to Claude Science
**Decision.** Enter the Build Track ("Build Beyond the Bench") with a provenance/contamination firewall,
not a biological finding.
**Why.** Claude Science already does reproducible artifacts, visualization, database skills, and citation
review — but does not prevent data leakage or cryptographically certify label provenance. That gap is our
target. Our real strength is data-integrity engineering, not domain biology; we chose the problem where
that engineering is the differentiator.

## DEC-1 — AI-assisted development boundary
**Decision.** Correctness-critical code is written directly and gated; delegable bulk (prose, boilerplate)
is routed to a heterogeneous OpenRouter "super squad" (e.g. DeepSeek) with a human/Claude gate.
**Why.** Explicitly permitted by the rules ("technical assistance", participant retains ownership) and the
Build Track requires Claude Code. Keeps cost down while preserving a review gate on anything that must be
correct. (In practice the application draft was squad-generated; the detector/manifest/tests were written
directly because they must be right.)
