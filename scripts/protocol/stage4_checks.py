#!/usr/bin/env python3
"""Plan Stage 4 checks for the windowed labeller. Writes reports/stage4_checks.{md,json}.

1. zero-offset test: 5 frames, R=100 m, prior = truth; >= 3/5 accepted_correct.
2. determinism: the zero-offset run repeated; rows identical except timing columns.
3. negative control: prior 2000 m east of truth, R=100 m; accepted_correct must be 0.
4. kill/resume: a run killed after its first frame, then resumed, equals an uninterrupted run
   and has no duplicate keys.
5. CPU cost per frame and per attempt.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL = ROOT / "scripts" / "label_windowed.py"
BASE = ROOT / "results" / "stage4" / "checks"
TIMING = {"runtime_s", "frame_setup_s", "frame_cpu_s"}
KEY = ("region", "filename", "variant", "radius_m", "offset_k")


def run(out: str, *extra: str) -> None:
    subprocess.run([sys.executable, str(LABEL), "--set", "stage4", "--out", out, *extra],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def rows(out: str) -> list[dict]:
    with (ROOT / out / "outcomes.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def strip(rs: list[dict]) -> list[tuple]:
    return sorted(tuple((k, v) for k, v in r.items() if k not in TIMING) for r in rs)


def count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def main() -> int:
    if BASE.exists():
        shutil.rmtree(BASE)
    BASE.mkdir(parents=True)
    rel = BASE.relative_to(ROOT).as_posix()
    res: dict = {}

    run(f"{rel}/zero_a", "--zero-offset", "--radii", "100")
    run(f"{rel}/zero_b", "--zero-offset", "--radii", "100")
    za, zb = rows(f"{rel}/zero_a"), rows(f"{rel}/zero_b")
    n_ok = sum(r["outcome_primary"] == "accepted_correct" for r in za)
    res["zero_offset"] = {
        "n": len(za),
        "accepted_correct": n_ok,
        "outcomes": {o: sum(r["outcome_primary"] == o for r in za) for o in ("accepted_correct", "rejected", "accepted_incorrect")},
        "per_frame": [(r["filename"], r["outcome_primary"], r["geodesic_error_m"], r["fine_inliers"]) for r in za],
        "pass": n_ok >= 3,
    }
    res["determinism"] = {"identical_excluding_timing": strip(za) == strip(zb), "pass": strip(za) == strip(zb)}

    run(f"{rel}/control", "--control-offset-m", "2000", "--radii", "100")
    ct = rows(f"{rel}/control")
    res["negative_control"] = {
        "n": len(ct),
        "truth_within_radius": sum(int(r["truth_within_radius"]) for r in ct),
        "outcomes": {o: sum(r["outcome_primary"] == o for r in ct) for o in ("accepted_correct", "rejected", "accepted_incorrect")},
        "pass": all(r["outcome_primary"] != "accepted_correct" for r in ct),
    }

    extra = ["--radii", "100,300", "--k", "2"]
    run(f"{rel}/resume_ref", *extra)
    part = ROOT / rel / "resume_cut" / "outcomes.csv"
    proc = subprocess.Popen([sys.executable, str(LABEL), "--set", "stage4", "--out", f"{rel}/resume_cut", *extra],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while count_rows(part) < 4 and proc.poll() is None and time.time() - t0 < 900:
        time.sleep(0.5)
    killed_at = count_rows(part)
    finished_before_kill = proc.poll() is not None
    proc.kill()
    proc.wait()
    run(f"{rel}/resume_cut", *extra, "--resume")
    ref, cut = rows(f"{rel}/resume_ref"), rows(f"{rel}/resume_cut")
    keys = [tuple(r[k] for k in KEY) for r in cut]
    res["kill_resume"] = {
        "rows_when_killed": killed_at,
        "process_finished_before_kill": finished_before_kill,
        "rows_after_resume": len(cut),
        "rows_reference": len(ref),
        "duplicate_keys": len(keys) - len(set(keys)),
        "identical_to_uninterrupted_excluding_timing": strip(ref) == strip(cut),
        "pass": (not finished_before_kill) and 0 < killed_at < len(ref)
                and len(keys) == len(set(keys)) and strip(ref) == strip(cut),
    }

    all_rows = za + ref
    frames = {}
    for r in all_rows:
        frames.setdefault((r["variant"], r["filename"]), float(r["frame_cpu_s"]))
    att = [float(r["runtime_s"]) for r in all_rows]
    by_r = {}
    for r in ref:
        by_r.setdefault(r["radius_m"], []).append(float(r["runtime_s"]))
    res["cost"] = {
        "mean_cpu_s_per_frame": round(sum(frames.values()) / len(frames), 2),
        "mean_wall_s_per_attempt": round(sum(att) / len(att), 3),
        "mean_wall_s_per_attempt_by_radius": {k: round(sum(v) / len(v), 3) for k, v in by_r.items()},
        "mean_frame_setup_wall_s": round(sum(float(r["frame_setup_s"]) for r in all_rows) / len(all_rows), 3),
        "note": "frame_cpu_s sums CPU over all threads; embedding extraction may have run concurrently",
    }
    res["all_pass"] = all(v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v)

    (ROOT / "reports" / "stage4_checks.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = ["# Stage 4 checks (windowed dinov2_c2f)", "", f"**All pass: {res['all_pass']}**", ""]
    for k, v in res.items():
        if isinstance(v, dict):
            md.append(f"## {k}")
            md.append("```json")
            md.append(json.dumps(v, indent=2))
            md.append("```")
    (ROOT / "reports" / "stage4_checks.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
