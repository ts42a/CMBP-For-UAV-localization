#!/usr/bin/env python3
"""Rerun the untouched Stage 3B validation gate with the re-frozen matcher.

Region 03 val_queries (30 frames, never used for tuning) from configs/stage3b_splits.yaml,
full-map search as in the original gate, pre-registered accept rule and tolerance.
Criteria are fixed in reports/verifier_switch_protocol.md.

  py -3 scripts/protocol/stage3b_val_gate_v2.py --run        # resumable outcomes
  py -3 scripts/protocol/stage3b_val_gate_v2.py --panels     # render review panels
  py -3 scripts/protocol/stage3b_val_gate_v2.py --decide     # needs reports/stage3b_gate_v2_visual_review.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "localization"))
from dino_embedder import DinoGlobalEmbedder, load_embedding_cache  # noqa: E402
from geo_utils import geodesic_m  # noqa: E402
from label_windowed import is_accepted  # noqa: E402
from patch_verify import PatchVerifier  # noqa: E402
from project_paths import CACHE_EMB, DATASET_ROOT  # noqa: E402
from query_select import load_manifest_region  # noqa: E402
from region_map import load_region  # noqa: E402
from tile_index import TileSpec  # noqa: E402
from windowed_matcher import QueryContext, RegionIndex, is_patch, localize_fullmap  # noqa: E402

PREREG = ROOT / "configs" / "preregistration_v1.yaml"
OUT = ROOT / "results" / "stage3b_gate_v2"
CSV_OUT = OUT / "val_outcomes.csv"
REVIEW = ROOT / "reports" / "stage3b_gate_v2_visual_review.csv"
FIELDS = ["filename", "num", "pred_lat", "pred_lon", "predicted_tile_id", "predicted_raster_id", "top_score",
          "score_margin", "n_candidates", "fine_inliers", "used_homography", "geo_verify_status",
          "patch_n_mutual", "patch_query_rot_deg", "patch_sim_scale", "patch_sim_angle_deg", "patch_context_scale",
          "region", "radius_m", "offset_k", "geodesic_error_m", "accepted", "outcome_primary",
          "search_used_gt_position", "runtime_s", "matcher_id", "config_sha256"]


def cfg() -> tuple[dict, dict, Path]:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    path = ROOT / prereg["inputs"]["frozen_matcher"]
    return prereg, yaml.safe_load(path.read_text(encoding="utf-8")), path


def val_frames() -> list[dict]:
    sp = yaml.safe_load((ROOT / "configs" / "stage3b_splits.yaml").read_text(encoding="utf-8"))
    names = [q["filename"] for q in sp["val_queries"]]
    by = {r["filename"]: r for r in load_manifest_region("03")}
    return [by[n] for n in names]


def run() -> None:
    prereg, frozen, path = cfg()
    rule, tol = prereg["accept_rule"]["selected"], float(prereg["tolerances_m"]["primary"])
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if CSV_OUT.is_file():
        done = {r["filename"] for r in csv.DictReader(CSV_OUT.open(encoding="utf-8", newline=""))}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rasters, tiles = load_region("03", TileSpec(1024, 512))
    embs, _, _ = load_embedding_cache(CACHE_EMB / "region03_dinov2_vits14_224.npz")
    idx = RegionIndex(tiles, rasters, embs)
    emb = DinoGlobalEmbedder(224)
    ver = PatchVerifier(emb.model, frozen["fine_match"]) if is_patch(frozen) else None
    new = not CSV_OUT.is_file()
    with CSV_OUT.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for q in val_frames():
            if q["filename"] in done:
                continue
            idx.clear_cache()
            bgr = cv2.imread(str(DATASET_ROOT / "03" / "drone" / q["filename"]))
            res = localize_fullmap(QueryContext(bgr, emb, frozen, ver), idx, frozen, ver)
            err = geodesic_m(float(q["lon"]), float(q["lat"]), float(res["pred_lon"]), float(res["pred_lat"]))
            row = {**res, "filename": q["filename"], "num": q["num"], "region": "03", "radius_m": "full_map",
                   "offset_k": "", "geodesic_error_m": round(err, 3), "search_used_gt_position": False,
                   "config_sha256": sha}
            acc = is_accepted(rule, row)
            row["accepted"] = int(acc)
            row["outcome_primary"] = "rejected" if not acc else ("accepted_correct" if err <= tol else "accepted_incorrect")
            w.writerow(row)
            f.flush()
            print(q["filename"], row["outcome_primary"], row["geodesic_error_m"], row["fine_inliers"], flush=True)


def panels() -> None:
    prereg, _, _ = cfg()
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "analysis" / "verify_panels.py"),
                           str(CSV_OUT.relative_to(ROOT)), "--rule", prereg["accept_rule"]["selected"],
                           "--label", "val_gate_v2", "--out", "figures/stage3b_gate_v2"])


def decide() -> int:
    prereg, frozen, path = cfg()
    rows = list(csv.DictReader(CSV_OUT.open(encoding="utf-8", newline="")))
    manifest = {r["filename"]: r for r in load_manifest_region("03")}
    n = len(rows)
    ac = sum(r["outcome_primary"] == "accepted_correct" for r in rows)
    ai = sum(r["outcome_primary"] == "accepted_incorrect" for r in rows)
    rej = sum(r["outcome_primary"] == "rejected" for r in rows)
    prec = ac / (ac + ai) if ac + ai else None
    csv_copy = sum(abs(float(r["pred_lat"]) - float(manifest[r["filename"]]["lat"])) < 1e-9 and
                   abs(float(r["pred_lon"]) - float(manifest[r["filename"]]["lon"])) < 1e-9 for r in rows)
    visual = None
    if REVIEW.is_file():
        rv = [r for r in csv.DictReader(REVIEW.open(encoding="utf-8", newline="")) if r["outcome"] == "accepted_correct"]
        if rv:
            yes = sum(r["scene_match"] == "yes" for r in rv)
            visual = {"panels": len(rv), "yes": yes, "fraction": round(yes / len(rv), 3)}
    checks = {
        "n_val_30": {"value": n, "pass": n == 30},
        "accepted_correct_ge_5": {"value": ac, "pass": ac >= 5},
        "precision_ge_0.80": {"value": None if prec is None else round(prec, 3), "pass": prec is not None and prec >= 0.80},
        "mix_present": {"value": {"rejected": rej, "accepted_incorrect": ai}, "pass": rej + ai >= 1},
        "search_used_gt_position_0": {"value": sum(r["search_used_gt_position"] != "False" for r in rows),
                                      "pass": all(r["search_used_gt_position"] == "False" for r in rows)},
        "exact_csv_copies_0": {"value": csv_copy, "pass": csv_copy == 0},
        "visual_ge_0.80": {"value": visual, "pass": visual is not None and visual["fraction"] >= 0.80},
    }
    gate = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    errs = sorted(float(r["geodesic_error_m"]) for r in rows if r["outcome_primary"] == "accepted_correct")
    out = {"gate": gate, "matcher_config": path.relative_to(ROOT).as_posix(),
           "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "accept_rule": prereg["accept_rule"]["selected"], "checks": checks,
           "accepted_correct_errors_m": errs,
           "reference_stage3b_akaze": "9/30 within 100 m counting unverified tile-centre fallbacks; 2/30 homography-verified"}
    (ROOT / "reports" / "stage3b_gate_v2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = ["# Stage 3B validation gate — rerun with re-frozen matcher", "",
          f"**Gate: {gate}**", "",
          f"Matcher `{out['matcher_config']}`, accept rule `{out['accept_rule']}`, region-03 val (30 frames, untouched), "
          "full-map search. Criteria from `reports/verifier_switch_protocol.md`.", "",
          "| Check | Value | Pass |", "|---|---|---|"]
    md += [f"| {k} | {v['value']} | {'yes' if v['pass'] else 'no'} |" for k, v in checks.items()]
    md += ["", f"Accepted-correct errors (m): {errs}", "",
           "For reference, the original Stage 3B AKAZE gate reported 9/30 within 100 m, but only 2 of those were "
           "homography-verified; the rest were unverified tile-centre fallbacks."]
    (ROOT / "reports" / "stage3b_gate_v2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0 if gate == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--panels", action="store_true")
    ap.add_argument("--decide", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    if a.panels:
        panels()
    if a.decide:
        return decide()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
