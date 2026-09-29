#!/usr/bin/env python3
"""Label-noise bound (predictor protocol v5, label_noise_bound).

Ground-truth error in parts of some tracks (e.g. region 06) makes some correct fixes score as
accepted_incorrect, and no label-free check finds them. This bounds their influence: the
unchanged evaluator is run twice on the same features, once with the registered label
(accepted_correct) and once with every accepted fix counted as a success (as if every wrong fix
were a ground-truth error). Headline macro metrics are reported side by side; a conclusion that
flips between the two is sensitive to ground-truth noise.

  py -3 scripts/predictor/label_noise_bound.py --protocol configs/predictor_protocol_v5.yaml \\
      --features <features dir> --out <out dir> --name <report name>
Writes <out>/{registered,upper}/ (evaluator outputs), reports/<name>_label_noise_bound.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
KEY = ["region", "filename", "variant", "radius_m", "offset_k"]
SHOW = ["auroc", "pr_auc", "brier", "log_loss", "ece"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/predictor_protocol_v5.yaml")
    ap.add_argument("--features", required=True)
    ap.add_argument("--outcomes", default=None, help="default: the protocol's inputs.outcomes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    proto = yaml.safe_load((ROOT / args.protocol).read_text(encoding="utf-8"))
    outcomes = ROOT / (args.outcomes or proto["inputs"]["outcomes"])
    feat = ROOT / args.features
    out = ROOT / args.out
    rep_dir = ROOT / args.reports_dir

    f = pd.read_csv(feat / "features.csv", dtype={"region": str, "group_7": str, "variant": str})
    o = pd.read_csv(outcomes, dtype={"region": str, "variant": str})[KEY + ["outcome_primary"]]
    j = f[KEY].merge(o, on=KEY, how="left", validate="one_to_one")
    if j["outcome_primary"].isna().any():
        raise SystemExit(f"{int(j['outcome_primary'].isna().sum())} feature rows have no outcome in {outcomes}")
    registered = (j["outcome_primary"] == "accepted_correct").astype(int).to_numpy()
    if (registered != f["label"].to_numpy()).any():
        raise SystemExit("features.csv label does not equal accepted_correct from the outcomes")
    upper = j["outcome_primary"].isin(["accepted_correct", "accepted_incorrect"]).astype(int)
    changed = int((upper.to_numpy() != registered).sum())

    alt = out / "features_upper"
    alt.mkdir(parents=True, exist_ok=True)
    f.assign(label=upper.to_numpy()).to_csv(alt / "features.csv", index=False)
    for extra in ("embeddings.npz", "run_manifest.json"):
        shutil.copy2(feat / extra, alt / extra)

    runs = {}
    for tag, fdir in (("registered", feat), ("upper", alt)):
        subprocess.check_call([sys.executable, str(ROOT / "scripts/predictor/evaluate_predictors.py"), "--protocol", args.protocol,
                               "--features", str(fdir), "--out", str(out / tag), "--name", f"{args.name}_{tag}",
                               "--reports-dir", str(out / tag), "--figures-dir", str(out / tag / "figures")], cwd=ROOT)
        runs[tag] = json.loads((out / tag / f"{args.name}_{tag}_summary.json").read_text(encoding="utf-8"))

    rows, flips = [], []
    for key in runs["registered"]["models"]:
        a, b = runs["registered"]["models"][key]["macro"], runs["upper"]["models"][key]["macro"]
        rows.append({"model|variant": key, **{f"{k}_registered": a[k] for k in SHOW}, **{f"{k}_upper": b[k] for k in SHOW}})

    def order(tag: str, metric: str, variant: str = "raw") -> list[str]:
        ms = {k.split("|")[0]: v["macro"][metric] for k, v in runs[tag]["models"].items() if k.endswith("|" + variant)}
        return sorted((m for m in ms if ms[m] is not None), key=lambda m: -ms[m])

    for metric in ("auroc",):
        if order("registered", metric) != order("upper", metric):
            flips.append(f"raw macro {metric} ranking of models: {order('registered', metric)} -> {order('upper', metric)}")
    h3 = {}
    for tag in runs:
        h3[tag] = {m: (v["brier"]["cal_v2_fe"] < v["brier"]["raw"] and v["log_loss"]["cal_v2_fe"] < v["log_loss"]["raw"])
                   for m, v in runs[tag].get("h3_macro", {}).items()}
    for m in h3.get("registered", {}):
        if h3["registered"][m] != h3["upper"].get(m):
            flips.append(f"H3 (fe better than raw on Brier and log loss) for {m}: {h3['registered'][m]} -> {h3['upper'].get(m)}")

    summary = {"protocol": args.protocol, "features": args.features, "outcomes": str(outcomes.relative_to(ROOT)) if outcomes.is_relative_to(ROOT) else str(outcomes),
               "rows": len(f), "labels_changed_by_upper_bound": changed,
               "prevalence_registered": float(registered.mean()), "prevalence_upper": float(upper.mean()),
               "headline_macro": rows, "h3": h3, "flips": flips}
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"{args.name}_label_noise_bound.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def fmt(x):
        return "—" if x is None else f"{x:.3f}"
    md = [f"# {args.name} — label-noise bound", "",
          f"Registered label = accepted_correct at 100 m; upper bound = every accepted fix counted as a success. "
          f"{changed} of {len(f)} rows change label (prevalence {registered.mean():.3f} → {upper.mean():.3f}). "
          f"Same protocol (`{args.protocol}`), features and evaluator; only the label differs.", "",
          "| model | variant | macro AUROC reg → upper | macro Brier reg → upper | macro log loss reg → upper |",
          "|---|---|---|---|---|"]
    for r in rows:
        m, v = r["model|variant"].split("|")
        md.append(f"| {m} | {v} | {fmt(r['auroc_registered'])} → {fmt(r['auroc_upper'])} | "
                  f"{fmt(r['brier_registered'])} → {fmt(r['brier_upper'])} | {fmt(r['log_loss_registered'])} → {fmt(r['log_loss_upper'])} |")
    md += ["", "**Conclusions that flip:** " + ("; ".join(flips) if flips else "none (model AUROC ranking and H3 direction unchanged).")]
    (rep_dir / f"{args.name}_label_noise_bound.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
