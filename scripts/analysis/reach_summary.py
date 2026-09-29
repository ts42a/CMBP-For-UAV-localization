#!/usr/bin/env python3
"""Effective reach of the frozen localizer on the dev region (Stage 9 preparation).

Region 03 tune queries (localizer-dev region only), radius 100 m, prior centre displaced
0, 250, 500, 750, 1000, 1500, 2000 m east of truth (scripts/label_windowed.py --zero-offset /
--control-offset-m). The nominal radius is 100 m, so any success beyond ~100 m comes from the
verifier's context crops (2x and 3x a ~304 m tile).

  py -3 scripts/analysis/reach_summary.py   -> reports/stage9_reach_r03.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "stage9_prep" / "reach_r03"


def main() -> int:
    rows = []
    for d in sorted(BASE.glob("off*"), key=lambda p: int(p.name[3:])):
        o = pd.read_csv(d / "outcomes.csv")
        off = int(d.name[3:])
        vc = o.outcome_primary.value_counts()
        rows.append({"offset_m": off, "attempts": len(o), "n_window_tiles_median": float(o.n_window_tiles.median()),
                     "accepted_correct": int(vc.get("accepted_correct", 0)), "accepted_incorrect": int(vc.get("accepted_incorrect", 0)),
                     "rejected": int(vc.get("rejected", 0)),
                     "success_rate": round(float((o.outcome_primary == "accepted_correct").mean()), 3)})
    t = pd.DataFrame(rows)
    md = ["# Effective reach of the frozen localizer — dev region 03", "",
          "`scripts/analysis/reach_summary.py`. Region 03 tune queries (24 frames, localizer-dev region only), nominal radius "
          "**100 m**, prior centre displaced east of truth by the offset below; frozen matcher and accept rule `patch_20`. "
          "Outputs: `results/stage9_prep/reach_r03/off<m>/`.", "",
          "| prior offset (m) | attempts | window tiles (median) | accepted-correct | accepted-incorrect | rejected | success rate |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in t.itertuples():
        md.append(f"| {r.offset_m} | {r.attempts} | {r.n_window_tiles_median:.0f} | {r.accepted_correct} | {r.accepted_incorrect} | "
                  f"{r.rejected} | {r.success_rate:.3f} |")
    md += ["", "## Reading", "",
           "- The reach beyond the nominal radius is short: success is 100% with the prior on the truth, drops sharply by "
           "250 m and is 0% from 500 m. The verifier only accepts when the ~500 m query footprint lies inside a 600–900 m "
           "crop centred on a window tile, which tolerates roughly 100–200 m between that tile centre and the truth.",
           "- Effective reach ≈ R + ~250 m. An earlier estimate in the project evaluation (R + 455 m, from the crop size "
           "alone) was wrong and is corrected here.",
           "- The flat pilot success-by-radius pattern is a design effect (prior σ = R/2 scales the offset with R, and a "
           "larger window still contains tiles near the truth), not long reach.",
           "- Dev region only; other regions' reach also depends on height above ground (footprint) and is not measured "
           "here, to avoid using test-region outcomes."]
    (ROOT / "reports" / "stage9_reach_r03.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
