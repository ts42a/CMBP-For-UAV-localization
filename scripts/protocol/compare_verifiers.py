#!/usr/bin/env python3
"""Apply reports/verifier_switch_protocol.md to the region-03 tune comparison runs.

A = 216 pre-registered windowed attempts; B = zero-offset R=100 m on all 24 tune frames.
Criterion 5 (visual) is read from reports/verifier_visual_review.csv when present.
Writes reports/verifier_comparison.{md,csv,json}.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from label_windowed import is_accepted  # noqa: E402

CMP = ROOT / "results" / "stage4" / "verifier_cmp"
TOL = 100.0
VARIANTS = ["s1", "s2", "s3", "s23"]
THRESHOLDS = [10, 15, 20, 30]
REVIEW = ROOT / "reports" / "verifier_visual_review.csv"


def load(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metrics(rows: list[dict], rule: str) -> dict:
    acc = [r for r in rows if is_accepted(rule, r)]
    cor = [r for r in acc if float(r["geodesic_error_m"]) <= TOL]
    errs = np.array([float(r["geodesic_error_m"]) for r in acc]) if acc else np.array([])
    return {
        "attempts": len(rows),
        "accepted": len(acc),
        "accepted_correct": len(cor),
        "accepted_incorrect": len(acc) - len(cor),
        "precision": round(len(cor) / len(acc), 4) if acc else None,
        "accepted_correct_rate": round(len(cor) / len(rows), 4),
        "frames_with_correct": len({r["filename"] for r in cor}),
        "median_err_accepted_m": round(float(np.median(errs)), 1) if len(errs) else None,
        "p90_err_accepted_m": round(float(np.percentile(errs, 90)), 1) if len(errs) else None,
        "median_err_correct_m": round(float(np.median([float(r["geodesic_error_m"]) for r in cor])), 1) if cor else None,
    }


def main() -> int:
    akaze_a = load(ROOT / "results" / "stage4" / "tune" / "outcomes.csv")
    akaze_b = load(CMP / "akaze_zero" / "outcomes.csv")
    table = []
    base = {"verifier": "akaze", "rule": "homography_12", **{f"A_{k}": v for k, v in metrics(akaze_a, "homography_12").items()},
            **{f"B_{k}": v for k, v in metrics(akaze_b, "homography_12").items()}}
    table.append(base)
    for v in VARIANTS:
        a = load(CMP / f"patch_{v}_prior" / "outcomes.csv")
        b = load(CMP / f"patch_{v}_zero" / "outcomes.csv")
        if len(a) != 216 or len(b) != 24:
            raise SystemExit(f"{v}: incomplete runs A={len(a)} B={len(b)}")
        for t in THRESHOLDS:
            rule = f"patch_{t}"
            table.append({"verifier": f"patch_{v}", "rule": rule,
                          **{f"A_{k}": x for k, x in metrics(a, rule).items()},
                          **{f"B_{k}": x for k, x in metrics(b, rule).items()}})

    eligible = [t for t in table[1:] if t["A_accepted_correct"] >= 5]
    sel = max(eligible, key=lambda t: (t["A_precision"], t["A_accepted_correct"])) if eligible else None

    visual = None
    if sel and REVIEW.is_file():
        rv = [r for r in load(REVIEW) if r["verifier"] == sel["verifier"] and r["rule"] == sel["rule"]
              and r["outcome"] == "accepted_correct"]
        if rv:
            ok = sum(r["scene_match"] == "yes" for r in rv)
            visual = {"panels": len(rv), "scene_match_yes": ok, "fraction": round(ok / len(rv), 3)}

    crit = None
    if sel:
        crit = {
            "1_accepted_correct_ge_2x_akaze": {"value": sel["A_accepted_correct"], "need": 2 * base["A_accepted_correct"],
                                              "pass": sel["A_accepted_correct"] >= 2 * base["A_accepted_correct"]},
            "2_precision_ge_0.90": {"value": sel["A_precision"], "pass": sel["A_precision"] >= 0.90},
            "3_frames_with_correct_ge_8": {"value": sel["A_frames_with_correct"], "pass": sel["A_frames_with_correct"] >= 8},
            "4_zero_offset_ge_12_of_24": {"value": sel["B_accepted_correct"], "pass": sel["B_accepted_correct"] >= 12},
            "5_visual_ge_0.90": {"value": visual, "pass": (visual is not None and visual["fraction"] >= 0.90)},
        }
    decision = "IMPROVES" if crit and all(c["pass"] for c in crit.values()) else (
        "PENDING_VISUAL" if crit and all(c["pass"] for k, c in crit.items() if not k.startswith("5")) and visual is None else "FAILS")

    with (ROOT / "reports" / "verifier_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    out = {"selected": sel and {"verifier": sel["verifier"], "rule": sel["rule"]}, "criteria": crit,
           "decision": decision, "akaze_baseline": base}
    (ROOT / "reports" / "verifier_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# Verifier comparison — region 03 tune frames only", "",
          "Protocol: `reports/verifier_switch_protocol.md` (committed before any run). Tolerance 100 m.",
          "A = 216 pre-registered windowed attempts (24 frames x radii 100/300/1000 m x 3 offsets). "
          "B = zero-offset, R = 100 m, 24 frames.", "",
          "| Verifier | Rule | A accepted | A acc. correct | A acc. incorrect | A precision | A frames w/ correct | "
          "A median err (accepted) | A p90 err | B acc. correct /24 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in table:
        md.append(f"| {t['verifier']} | {t['rule']} | {t['A_accepted']} | {t['A_accepted_correct']} | {t['A_accepted_incorrect']} | "
                  f"{t['A_precision']} | {t['A_frames_with_correct']} | {t['A_median_err_accepted_m']} | "
                  f"{t['A_p90_err_accepted_m']} | {t['B_accepted_correct']} |")
    md += ["", f"**Selected by the pre-registered criterion:** `{sel['verifier']}` with `{sel['rule']}`" if sel else "**No setting eligible.**", ""]
    if crit:
        md += ["| Criterion | Value | Pass |", "|---|---|---|"]
        for k, c in crit.items():
            val = c["value"] if "need" not in c else f"{c['value']} (need {c['need']})"
            md.append(f"| {k} | {val} | {'yes' if c['pass'] else 'no'} |")
    md += ["", f"**Decision: {decision}**"]
    (ROOT / "reports" / "verifier_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
