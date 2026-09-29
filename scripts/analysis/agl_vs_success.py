#!/usr/bin/env python3
"""EXPLORATORY (post hoc): Stage 5 pilot success rate against height above ground.

Motivates the diagnostic model M4 (proposed, predictor protocol v5); it is not a test of any
registered hypothesis. Two forms were looked at, in this order, and both are reported:
(1) |log(AGL / AGL of dev region 03)| (the verifier's crop scales were tuned on region 03), and
(2) AGL itself. Region 07 excluded (D2).

  py -3 scripts/analysis/agl_vs_success.py      -> reports/agl_vs_success_exploratory.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    o = pd.read_csv(ROOT / "results/stage5/pilot/outcomes.csv", dtype={"region": str})
    a = pd.read_csv(ROOT / "datasets/manifests/frame_agl.csv", dtype={"region": str})
    o = o.merge(a[["region", "filename", "agl_m"]], on=["region", "filename"], validate="many_to_one")
    o = o[o.region != "07"]
    o["y"] = (o.outcome_primary == "accepted_correct").astype(int)
    reg = o.groupby("region").agg(success=("y", "mean"), agl_median=("agl_m", "median"), rows=("y", "size"))
    a03 = reg.loc["03", "agl_median"]
    reg["agl_ratio_to_03"] = reg.agl_median / a03
    fr = o.groupby(["region", "filename"]).agg(y=("y", "mean"), agl=("agl_m", "first")).reset_index()

    tests = {
        "(1) region: success vs -abs(log(AGL/AGL03))": spearmanr(reg.success, -np.abs(np.log(reg.agl_ratio_to_03))),
        "(1) frame: success vs -abs(log(AGL/AGL03))": spearmanr(fr.y, -np.abs(np.log(fr.agl / a03))),
        "(2) region: success vs AGL": spearmanr(reg.success, reg.agl_median),
        "(2) frame: success vs AGL": spearmanr(fr.y, fr.agl),
    }
    within = {r: spearmanr(g.y, g.agl).statistic for r, g in fr.groupby("region") if g.y.nunique() > 1 and g.agl.nunique() > 1}

    md = ["# Pilot success vs height above ground — EXPLORATORY, post hoc", "",
          "**Not a registered analysis.** Looked at after the pilot outcomes were known, to decide whether a flight-context "
          "diagnostic (M4) is worth registering. Two functional forms were tried, in the order below; both are reported. "
          "AGL from `reports/agl_height_audit.md` (CSV height − SRTM terrain). Region 07 excluded (D2). 10 frames per region, "
          "9 attempts per frame.", "",
          "| region | success (accepted-correct rate) | AGL median (m) | AGL / region 03 |", "|---|---:|---:|---:|"]
    for r, x in reg.sort_values("agl_median").iterrows():
        md.append(f"| {r} | {x.success:.3f} | {x.agl_median:.0f} | {x.agl_ratio_to_03:.2f} |")
    md += ["", "| test (Spearman) | rho | p | n |", "|---|---:|---:|---:|"]
    for k, t in tests.items():
        n = len(reg) if k.split(":")[0].endswith("region") else len(fr)
        md.append(f"| {k} | {t.statistic:.2f} | {t.pvalue:.3g} | {n} |")
    md += ["", "Within-region frame-level rho (AGL varies little inside a flight): "
           + ", ".join(f"{r} {v:.2f}" for r, v in within.items()), "",
           "## Reading", "",
           "- Across regions, success rises with height above ground (form 2); the region-03-centred form (1) is weaker. "
           "Higher flights give a larger ground footprint, closer to the verifier's 600–900 m context crops.",
           "- Inside a flight AGL barely varies and the relation is mixed, so AGL speaks to the **between-region base rate** "
           "(40% of label variance), which a map-only predictor cannot see.",
           "- Ten regions, post hoc, and confounded with terrain type (flat eastern plains fly higher above ground here). "
           "This motivates a diagnostic, not a claim."]
    out = ROOT / "reports" / "agl_vs_success_exploratory.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
