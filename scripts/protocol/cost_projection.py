#!/usr/bin/env python3
"""Stage 10 cost sheet: project the remaining workload from measured per-attempt timings.

Timing comes from any windowed outcomes file (PC pilot rows, or a Kaggle benchmark run):
per-frame setup (query features) + per-attempt wall time. Workload (configs/stage10_benchmark.yaml):
  full labels   = frames (all regions except 07) x 9 attempts
  H1 synthetic  = the same count (matched sample counts)
  replay        = sum over budgets of (budget x route frames) x drift levels x seeds x policies,
                  with query features cached per frame (assumption until Stage 9 is designed)
plus per-session overhead, times the safety factor.

  py -3 scripts/protocol/cost_projection.py --timing results/stage5/pilot/outcomes.csv --label pc_cpu
  py -3 scripts/protocol/cost_projection.py --timing results/stage10/kaggle_bench/outcomes.csv --label kaggle_gpu --runner-timing results/stage10/kaggle_bench/timing.json
Writes reports/stage10_cost_sheet_<label>.{md,json}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", required=True, help="outcomes.csv with runtime_s and frame_setup_s")
    ap.add_argument("--label", required=True)
    ap.add_argument("--runner-timing", default=None, help="timing.json from kaggle/stage10_runner.py (session overhead)")
    ap.add_argument("--weekly-quota-h", type=float, default=30.0, help="your Kaggle weekly GPU quota (confirm on your account)")
    ap.add_argument("--session-cap-h", type=float, default=9.0, help="lower of the quoted 9 h / 12 h caps")
    ap.add_argument("--drift-levels", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--policies", type=int, default=5, help="attempting policies (equal, random, texture, uncalibrated, calibrated)")
    ap.add_argument("--budgets", default="0.05,0.1,0.2,0.4", help="fractions of route frames attempted")
    args = ap.parse_args()

    bench = yaml.safe_load((ROOT / "configs" / "stage10_benchmark.yaml").read_text(encoding="utf-8"))
    safety = float(bench["projection"]["safety_factor"])
    excluded = set(yaml.safe_load((ROOT / "configs" / "predictor_protocol_v3.yaml").read_text(encoding="utf-8"))["inputs"]["excluded_regions"])
    with (ROOT / "datasets" / "manifests" / "manifest.csv").open(encoding="utf-8", newline="") as f:
        frames = sum(1 for r in csv.DictReader(f) if r["region_folder"] not in excluded)

    with (ROOT / args.timing).open(encoding="utf-8", newline="") as f:
        rs = list(csv.DictReader(f))
    attempt_s = float(np.mean([float(r["runtime_s"]) for r in rs]))
    setup = {}
    for r in rs:
        setup.setdefault((r["region"], r["filename"]), float(r["frame_setup_s"]))
    setup_s = float(np.mean(list(setup.values())))
    frame_s = setup_s + 9 * attempt_s

    overhead_s = 180.0
    overhead_src = "assumed 180 s (model load + data access) per session"
    if args.runner_timing:
        t = json.loads((ROOT / args.runner_timing).read_text(encoding="utf-8"))
        overhead_s = float(t["import_torch_s"]) + float(t["model_load_s"])
        overhead_src = f"measured import + model load {overhead_s:.0f} s per session (data access not included)"

    budgets = [float(b) for b in args.budgets.split(",")]
    replay_attempts = sum(b * frames for b in budgets) * args.drift_levels * args.seeds * args.policies
    items = {
        "full_labels": {"units": f"{frames} frames x 9 attempts", "hours": frames * frame_s / 3600},
        "h1_synthetic": {"units": f"{frames} synthetic queries x 9 attempts", "hours": frames * frame_s / 3600},
        "replay": {"units": (f"{replay_attempts:,.0f} attempts = sum(budgets {args.budgets}) x {frames} frames x "
                             f"{args.drift_levels} drift x {args.seeds} seeds x {args.policies} policies"),
                   "hours": (replay_attempts * attempt_s + frames * setup_s) / 3600},
    }
    compute_h = sum(v["hours"] for v in items.values())
    sessions = math.ceil(compute_h / args.session_cap_h)
    overhead_h = sessions * overhead_s / 3600
    total_h = (compute_h + overhead_h) * safety
    quota_weeks = total_h / args.weekly_quota_h
    out = {
        "label": args.label, "timing_source": args.timing, "n_timing_rows": len(rs),
        "attempt_wall_mean_s": round(attempt_s, 3), "frame_setup_mean_s": round(setup_s, 3),
        "frame_total_s": round(frame_s, 2), "frames_full_run": frames,
        "items": {k: {"units": v["units"], "hours": round(v["hours"], 1)} for k, v in items.items()},
        "sessions": sessions, "overhead_hours": round(overhead_h, 2), "overhead_source": overhead_src,
        "safety_factor": safety, "total_hours_with_safety": round(total_h, 1),
        "weekly_quota_h": args.weekly_quota_h, "quota_weeks": round(quota_weeks, 2),
        "stay_on_kaggle_rule": bench["projection"]["stay_on_kaggle_if"],
        "stay_on_kaggle": quota_weeks <= 3.0,
        "labels_plus_h1_hours_with_safety": round((items["full_labels"]["hours"] + items["h1_synthetic"]["hours"]) * safety, 1),
    }
    (ROOT / "reports" / f"stage10_cost_sheet_{args.label}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = [f"# Stage 10 cost sheet — {args.label}", "",
          f"Timing from `{args.timing}` ({len(rs)} attempts): attempt {attempt_s:.2f} s, frame setup {setup_s:.2f} s, "
          f"frame total {frame_s:.1f} s (wall).", "",
          "| item | workload | hours |", "|---|---|---:|"]
    md += [f"| {k} | {v['units']} | {v['hours']:.1f} |" for k, v in items.items()]
    md += [f"| session overhead | {sessions} sessions ({overhead_src}) | {overhead_h:.1f} |",
           f"| **total x {safety} safety** | | **{total_h:.1f}** |", "",
           f"Labels + H1 only (x {safety}): {out['labels_plus_h1_hours_with_safety']} h.", "",
           f"Weekly quota assumed {args.weekly_quota_h:g} h (**confirm on your account**) → {quota_weeks:.2f} quota-weeks. "
           f"Rule: {out['stay_on_kaggle_rule']} → **{'stay on Kaggle' if out['stay_on_kaggle'] else 'exceeds 3 quota-weeks'}**.", "",
           "Replay counts are assumptions until Stage 9 fixes routes, drift levels, seeds, policies and budgets; "
           "they assume query features are cached per frame and reused across policies and seeds."]
    (ROOT / "reports" / f"stage10_cost_sheet_{args.label}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())  # Windows consoles are often cp1252
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
