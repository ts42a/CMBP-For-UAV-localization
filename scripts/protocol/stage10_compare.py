#!/usr/bin/env python3
"""Stage 10: compare a benchmark run with the PC reference rows (configs/stage10_benchmark.yaml).

Applies the frozen agreement criteria to the attempts both runs share, lists every
disagreement, and summarises timings (attempt wall time by radius, per-frame setup) for the
run and for the PC reference on the same keys.

  py -3 scripts/protocol/stage10_compare.py results/stage10/kaggle_bench --name stage10_benchmark
Writes reports/<name>.{md,json}.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))
from geo_utils import geodesic_m  # noqa: E402

KEY = ("region", "filename", "variant", "radius_m", "offset_k")
MAX_OUTCOME_DIFF_FRAC = 0.03   # >= 97% identical outcome_primary
POS_TOL_M = 5.0
POS_FRAC = 0.95


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def timing(rs: list[dict]) -> dict:
    by_r = {}
    for R in sorted({int(r["radius_m"]) for r in rs}):
        v = np.array([float(r["runtime_s"]) for r in rs if int(r["radius_m"]) == R])
        by_r[str(R)] = {"median_s": round(float(np.median(v)), 3), "mean_s": round(float(v.mean()), 3)}
    frames = {}
    for r in rs:
        frames.setdefault((r["region"], r["filename"]), float(r["frame_setup_s"]))
    per_frame = {}
    for r in rs:
        k = (r["region"], r["filename"])
        per_frame[k] = per_frame.get(k, frames[k]) + float(r["runtime_s"])
    pf = np.array(list(per_frame.values()))
    return {"attempt_wall_by_radius": by_r, "attempt_wall_mean_s": round(float(np.mean([float(r["runtime_s"]) for r in rs])), 3),
            "frame_setup_mean_s": round(float(np.mean(list(frames.values()))), 3),
            "frame_total_mean_s": round(float(pf.mean()), 2), "frame_total_median_s": round(float(np.median(pf)), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--reference", default="results/stage5/pilot/outcomes.csv")
    ap.add_argument("--expect-frames", type=int, default=20)
    ap.add_argument("--name", default="stage10_benchmark")
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    run = rows(run_dir / "outcomes.csv")
    ref = {tuple(r[k] for k in KEY): r for r in rows(ROOT / args.reference)}
    keys = [tuple(r[k] for k in KEY) for r in run]
    n_frames = len({(r["region"], r["filename"]) for r in run})
    missing = [k for k in keys if k not in ref]
    checks = {
        "n_attempts": len(run), "n_frames": n_frames, "expected_frames": args.expect_frames,
        "expected_attempts": args.expect_frames * 9,
        "no_duplicate_keys": len(set(keys)) == len(keys),
        "all_keys_in_reference": not missing,
        "config_sha256_matches_reference": {r["config_sha256"] for r in run} == {ref[k]["config_sha256"] for k in keys if k in ref},
    }
    checks["complete"] = checks["n_attempts"] == checks["expected_attempts"] and checks["n_frames"] == args.expect_frames

    diffs, pos = [], []
    for r, k in zip(run, keys):
        p = ref.get(k)
        if p is None:
            continue
        same = r["outcome_primary"] == p["outcome_primary"]
        d_m = None
        if r["pred_lat"] and p["pred_lat"]:
            d_m = geodesic_m(float(r["pred_lon"]), float(r["pred_lat"]), float(p["pred_lon"]), float(p["pred_lat"]))
            if r["accepted"] == "1" and p["accepted"] == "1":
                pos.append(d_m)
        if not same or (d_m is not None and d_m > POS_TOL_M and r["accepted"] == "1"):
            diffs.append({"key": dict(zip(KEY, k)), "run_outcome": r["outcome_primary"], "ref_outcome": p["outcome_primary"],
                          "run_inliers": r["fine_inliers"], "ref_inliers": p["fine_inliers"],
                          "position_diff_m": None if d_m is None else round(d_m, 2)})
    n_out_diff = sum(d["run_outcome"] != d["ref_outcome"] for d in diffs)
    pos_ok = sum(x <= POS_TOL_M for x in pos)
    checks.update({
        "outcome_disagreements": n_out_diff,
        "outcome_agreement_frac": round(1 - n_out_diff / len(run), 4) if run else None,
        "outcome_criterion_pass": n_out_diff <= MAX_OUTCOME_DIFF_FRAC * len(run),
        "both_accepted": len(pos), "position_within_5m": pos_ok,
        "position_max_diff_m": round(max(pos), 3) if pos else None,
        "position_criterion_pass": (pos_ok >= POS_FRAC * len(pos)) if pos else True,
    })
    checks["agreement_pass"] = all(checks[c] for c in ("complete", "no_duplicate_keys", "all_keys_in_reference",
                                                       "config_sha256_matches_reference", "outcome_criterion_pass",
                                                       "position_criterion_pass"))

    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8")) if (run_dir / "env.json").is_file() else None
    tim = json.loads((run_dir / "timing.json").read_text(encoding="utf-8")) if (run_dir / "timing.json").is_file() else None
    summary = {"run_dir": args.run_dir, "checks": checks, "disagreements": diffs,
               "timing_run": timing(run), "timing_pc_reference_same_keys": timing([ref[k] for k in keys if k in ref]),
               "env": env, "runner_timing": tim}
    (ROOT / "reports" / f"{args.name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    t_run, t_pc = summary["timing_run"], summary["timing_pc_reference_same_keys"]
    md = [f"# {args.name} — Stage 10 benchmark vs PC reference", "",
          f"Run `{args.run_dir}` ({env['gpu'] if env and env.get('gpu') else 'CPU'}; bundle `{env['bundle_commit'][:7] if env else '?'}`) "
          f"vs `{args.reference}` on the same attempt keys. Criteria frozen in `configs/stage10_benchmark.yaml`.", "",
          f"**Agreement: {'PASS' if checks['agreement_pass'] else 'FAIL'}**", "", "```json", json.dumps(checks, indent=2), "```", "",
          "## Timing (wall seconds)", "",
          "| | run | PC reference |", "|---|---:|---:|",
          f"| attempt mean | {t_run['attempt_wall_mean_s']} | {t_pc['attempt_wall_mean_s']} |",
          *[f"| attempt median, R={R} m | {t_run['attempt_wall_by_radius'][R]['median_s']} | {t_pc['attempt_wall_by_radius'][R]['median_s']} |"
            for R in t_run["attempt_wall_by_radius"]],
          f"| frame setup mean | {t_run['frame_setup_mean_s']} | {t_pc['frame_setup_mean_s']} |",
          f"| frame total mean (setup + 9 attempts) | {t_run['frame_total_mean_s']} | {t_pc['frame_total_mean_s']} |", ""]
    if tim:
        md += ["Runner: " + json.dumps(tim), ""]
    md += ["## Disagreements", ""]
    md += ["None."] if not diffs else ["| key | run | PC | inliers run/PC | position diff m |", "|---|---|---|---|---:|"] + [
        f"| {d['key']['filename']} R={d['key']['radius_m']} k={d['key']['offset_k']} | {d['run_outcome']} | {d['ref_outcome']} | "
        f"{d['run_inliers']}/{d['ref_inliers']} | {d['position_diff_m']} |" for d in diffs]
    (ROOT / "reports" / f"{args.name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    return 0 if checks["agreement_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
