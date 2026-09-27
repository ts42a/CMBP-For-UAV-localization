#!/usr/bin/env python3
"""Apply the pre-registered accept-rule selection criterion to region-03 tune outcomes.

Criterion (configs/preregistration_v1.yaml): highest precision at the primary tolerance
among candidates with >= 5 accepted_correct attempts; ties -> more accepted_correct.
Writes reports/accept_rule_selection.{csv,md}. Does not edit the pre-registration.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from label_windowed import is_accepted  # noqa: E402

PREREG = ROOT / "configs" / "preregistration_v1.yaml"
TUNE = ROOT / "results" / "stage4" / "tune" / "outcomes.csv"
OUT_CSV = ROOT / "reports" / "accept_rule_selection.csv"
OUT_MD = ROOT / "reports" / "accept_rule_selection.md"
MIN_CORRECT = 5


def main() -> int:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    tol = float(prereg["tolerances_m"]["primary"])
    rows = list(csv.DictReader(TUNE.open(encoding="utf-8", newline="")))
    if {r["region"] for r in rows} != {"03"}:
        raise SystemExit("tune outcomes must be region 03 only")

    table = []
    for rule in prereg["accept_rule"]["candidates"]:
        for radius in ["all"] + sorted({r["radius_m"] for r in rows}, key=int):
            sub = rows if radius == "all" else [r for r in rows if r["radius_m"] == radius]
            acc = [r for r in sub if is_accepted(rule, r)]
            corr = [r for r in acc if float(r["geodesic_error_m"]) <= tol]
            table.append({
                "rule": rule,
                "radius_m": radius,
                "attempts": len(sub),
                "accepted": len(acc),
                "accepted_correct": len(corr),
                "accepted_incorrect": len(acc) - len(corr),
                "rejected": len(sub) - len(acc),
                "precision": round(len(corr) / len(acc), 4) if acc else "",
                "accepted_correct_rate": round(len(corr) / len(sub), 4) if sub else "",
            })

    pooled = [t for t in table if t["radius_m"] == "all"]
    eligible = [t for t in pooled if t["accepted_correct"] >= MIN_CORRECT]
    selected = None
    if eligible:
        selected = max(eligible, key=lambda t: (t["precision"], t["accepted_correct"]))["rule"]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    lines = [
        "# Accept-rule selection (region 03 tune set only)",
        "",
        f"Data: `results/stage4/tune/outcomes.csv` — {len(rows)} windowed attempts, "
        f"{len({r['filename'] for r in rows})} frames. Primary tolerance {tol:g} m.",
        f"Criterion: highest precision among rules with >= {MIN_CORRECT} accepted-correct; ties -> more accepted-correct.",
        "",
        "| Rule | Attempts | Accepted | Acc. correct | Acc. incorrect | Precision | Acc.-correct rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for t in pooled:
        lines.append(f"| {t['rule']} | {t['attempts']} | {t['accepted']} | {t['accepted_correct']} | "
                     f"{t['accepted_incorrect']} | {t['precision']} | {t['accepted_correct_rate']} |")
    lines += ["", f"**Selected: `{selected}`**" if selected else "**No rule met the minimum; none selected.**",
              "", "Per-radius breakdown: `reports/accept_rule_selection.csv`."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
