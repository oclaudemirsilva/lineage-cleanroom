"""lineage_cleanroom/report.py — escreve o relatório reprodutível em disco (SRP: só I/O de saída).

Dois artefatos por auditoria:
  - `<stem>.manifest.json` — o manifesto ASSINADO (anexável a paper/submissão/model card).
  - `<stem>.report.md`     — relatório humano legível do veredito.
"""
from __future__ import annotations

import json
from pathlib import Path

from .pipeline import AuditResult


def write_report(result: AuditResult, out_dir: str | Path, *, stem: str = "lineage_cleanroom") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"{stem}.manifest.json"
    report_path = out / f"{stem}.report.md"

    manifest_path.write_text(json.dumps(result.manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_render_md(result), encoding="utf-8")
    return {"manifest": str(manifest_path), "report": str(report_path)}


def _render_md(r: AuditResult) -> str:
    leak = r.leak_report
    prov = r.provenance_report
    status = "✅ CLEAN" if r.passed else "❌ CONTAMINATION"
    lines = [
        "# Lineage CleanRoom — audit report",
        "",
        f"**Status:** {status}",
        f"**Verdict:** {r.verdict}",
        f"**Signed manifest verifies:** {r.verify_ok}",
        f"**Split:** {r.n_train} train / {r.n_test} test rows",
        "",
        "## Leakage gate (features)",
        f"- test rows with an exact feature duplicate in train: **{leak.get('feature_dup_test_rows')}**",
        f"- test leak fraction: **{leak.get('test_leak_fraction')}**",
        f"- groups spanning train/test: **{leak.get('group_span_test_rows')}**",
        "",
        "## Provenance gate (label origin)",
    ]
    if not prov.get("provided"):
        lines.append(f"- {prov.get('note', 'not provided')}")
    else:
        lines.append(f"- human fraction in eval set: **{prov.get('human_fraction_test')}**")
        lines.append(f"- label sources (test): {prov.get('dist_test')}")
        if prov.get("violations"):
            lines.append("- violations:")
            lines += [f"  - {v}" for v in prov["violations"]]
        else:
            lines.append("- no provenance violations")
    lines += [
        "",
        "## Provenance manifest (Ed25519, tamper-evident)",
        "The `.manifest.json` next to this file is cryptographically signed. Any change to a single",
        "byte of its content invalidates the signature — attach it to a paper or submission as a",
        "reproducibility certificate of this dataset's lineage and integrity verdict.",
    ]
    return "\n".join(lines) + "\n"
