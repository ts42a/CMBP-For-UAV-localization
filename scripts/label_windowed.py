#!/usr/bin/env python3
"""Windowed real-outcome labelling (plan Stages 4-5) with the frozen dinov2_c2f matcher.

Ground truth is used only to place the prior centre (truth + pre-registered offset)
and to score error afterwards. Resumable: rows are appended one at a time and keys
already on disk are skipped.

Examples:
  py -3 scripts/label_windowed.py --set tune --out results/stage4/tune
  py -3 scripts/label_windowed.py --set stage4 --zero-offset --radii 100 --out results/stage4/zero
  py -3 scripts/label_windowed.py --set pilot --per-region 10 --out results/stage5/pilot --resume
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))

from dino_embedder import DinoGlobalEmbedder, load_embedding_cache  # noqa: E402
from patch_verify import PatchVerifier  # noqa: E402
from geo_utils import geodesic_m  # noqa: E402
from project_paths import CACHE_EMB, DATASET_ROOT  # noqa: E402
from query_select import load_manifest_region, select_queries  # noqa: E402
from region_map import load_region  # noqa: E402
from tile_index import TileSpec  # noqa: E402
from windowed_matcher import (  # noqa: E402
    M_PER_DEG_LAT,
    M_PER_DEG_LON_EQ,
    QueryContext,
    RegionIndex,
    is_patch,
    localize_windowed,
)

PREREG = ROOT / "configs" / "preregistration_v1.yaml"
SPLITS = ROOT / "configs" / "splits_v1.json"
STAGE3B_SPLITS = ROOT / "configs" / "stage3b_splits.yaml"
REGIONS = [f"{i:02d}" for i in range(1, 12)]

FIELDS = [
    "region", "group_7", "filename", "num", "variant", "radius_m", "offset_k",
    "offset_east_m", "offset_north_m", "offset_dist_m", "center_lat", "center_lon",
    "truth_dist_to_center_m", "truth_within_radius", "n_window_tiles", "n_candidates",
    "pred_lat", "pred_lon", "predicted_tile_id", "predicted_raster_id", "top_score",
    "score_margin", "fine_inliers", "used_homography", "geo_verify_status",
    "geodesic_error_m", "accept_rule", "accepted", "outcome_primary",
    "patch_n_mutual", "patch_query_rot_deg", "patch_sim_scale", "patch_sim_angle_deg", "patch_context_scale",
    "runtime_s", "frame_setup_s", "frame_cpu_s", "matcher_id", "config_path", "config_sha256", "prereg_sha256",
]
KEY = ("region", "filename", "variant", "radius_m", "offset_k")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--", "scripts", "configs"], cwd=ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return True


def region_to_group7() -> dict[str, str]:
    s = json.loads(SPLITS.read_text(encoding="utf-8"))["schemes"]["primary_7group"]["groups"]
    return {r: g for g, v in s.items() for r in v["regions"]}


def is_accepted(rule: str | None, row: dict) -> bool | None:
    if rule is None:
        return None
    if row["geo_verify_status"] == "empty_window" or row["pred_lat"] == "":
        return False
    if rule == "any_output":
        return True
    kind, _, n = rule.rpartition("_")
    if kind not in ("homography", "patch") or not n.isdigit():
        raise ValueError(f"unknown accept rule {rule}")
    status = {"homography": "homography", "patch": "patch_similarity"}[kind]
    return (int(row["used_homography"]) == 1 and row["geo_verify_status"] == status
            and int(row["fine_inliers"]) >= int(n))


def offset_for(seed: int, region: str, filename: str, radius: int, k: int) -> tuple[float, float]:
    h = hashlib.sha256(f"{seed}|{region}|{filename}|{radius}|{k}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    e, n = rng.normal(0.0, 0.5 * radius, size=2)
    return float(e), float(n)


def shift(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    return (
        lat + north_m / M_PER_DEG_LAT,
        lon + east_m / (M_PER_DEG_LON_EQ * np.cos(np.radians(lat))),
    )


def pick_frames(args) -> list[dict]:
    if args.set in ("tune", "stage4"):
        sp = yaml.safe_load(STAGE3B_SPLITS.read_text(encoding="utf-8"))
        names = [q["filename"] for q in sp["tune_queries"]]
        by_name = {r["filename"]: r for r in load_manifest_region(sp["region"])}
        frames = [by_name[n] for n in names]
        if args.set == "stage4":
            idx = np.linspace(0, len(frames) - 1, args.n_frames).round().astype(int)
            frames = [frames[i] for i in idx]
        return frames
    regions = REGIONS if args.regions == "all" else [r.strip() for r in args.regions.split(",")]
    frames = []
    for r in regions:
        m = load_manifest_region(r)
        frames += select_queries(m, min(args.per_region, len(m)), seed=args.select_seed)
    return frames


def done_keys(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {tuple(r[k] for k in KEY) for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["tune", "stage4", "pilot"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--radii", default=None, help="comma list; default = pre-registered radii")
    ap.add_argument("--k", type=int, default=None, help="offsets per radius; default = pre-registered")
    ap.add_argument("--zero-offset", action="store_true")
    ap.add_argument("--control-offset-m", type=float, default=None, help="fixed eastward offset (negative control)")
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--regions", default="all")
    ap.add_argument("--per-region", type=int, default=10)
    ap.add_argument("--select-seed", type=int, default=20260927)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--matcher-config", default=None, help="default = preregistration inputs.frozen_matcher")
    ap.add_argument("--accept-rule", default=None, help="override the pre-registered rule; 'none' leaves outcomes blank")
    args = ap.parse_args()

    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    cfg_path = ROOT / (args.matcher_config or prereg["inputs"]["frozen_matcher"])
    frozen = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    radii = [int(x) for x in args.radii.split(",")] if args.radii else list(prereg["search_window"]["radii_m"])
    k_per = 1 if (args.zero_offset or args.control_offset_m is not None) else (args.k or prereg["prior_offset"]["offsets_per_frame_per_radius"])
    seed = int(prereg["prior_offset"]["seed"])
    tol = float(prereg["tolerances_m"]["primary"])
    rule = prereg["accept_rule"]["selected"] if args.accept_rule is None else (None if args.accept_rule == "none" else args.accept_rule)
    variant = "zero" if args.zero_offset else (f"control{int(args.control_offset_m)}" if args.control_offset_m is not None else "prior")
    g7 = region_to_group7()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "outcomes.csv"
    if out_csv.exists() and not args.resume:
        raise SystemExit(f"{out_csv} exists; pass --resume or choose a new --out")
    done = done_keys(out_csv)

    frames = pick_frames(args)
    if args.max_frames:
        frames = frames[: args.max_frames]
    cfg_sha, prereg_sha = sha256(cfg_path), sha256(PREREG)
    cfg_rel = cfg_path.relative_to(ROOT).as_posix()
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "git_dirty_scripts_or_configs": git_dirty(),
        "config_path": cfg_rel,
        "config_sha256": cfg_sha,
        "prereg_sha256": prereg_sha,
        "splits_sha256": sha256(SPLITS),
        "accept_rule": rule,
        "args": vars(args),
        "radii_m": radii,
        "offsets_per_radius": k_per,
        "variant": variant,
        "n_frames": len(frames),
    }, indent=2), encoding="utf-8")

    spec = TileSpec(int(frozen["retrieval"]["tiling"]["tile_size_px"]), int(frozen["retrieval"]["tiling"]["stride_px"]))
    embedder = DinoGlobalEmbedder(image_size=int(frozen["model"]["image_size"]))
    verifier = PatchVerifier(embedder.model, frozen["fine_match"]) if is_patch(frozen) else None
    new_rows = 0
    f_out = out_csv.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f_out, fieldnames=FIELDS)
    if out_csv.stat().st_size == 0:
        writer.writeheader()
        f_out.flush()

    current_region, index = None, None
    for q in frames:
        region = q["region_folder"]
        wanted = [(r, k) for r in radii for k in range(k_per)
                  if (region, q["filename"], variant, str(r), str(k)) not in done]
        if not wanted:
            continue
        if region != current_region:
            rasters, tiles = load_region(region, spec)
            cache = CACHE_EMB / f"region{region}_dinov2_vits14_{int(frozen['model']['image_size'])}.npz"
            embs, _, _ = load_embedding_cache(cache)
            if len(embs) != len(tiles):
                raise RuntimeError(f"region {region}: {len(embs)} embeddings != {len(tiles)} tiles")
            index = RegionIndex(tiles, rasters, embs)
            current_region = region
            print(f"region {region}: {len(tiles)} tiles", flush=True)
        index.clear_cache()

        path = Path(q["image_path"]) if q.get("image_path") else DATASET_ROOT / region / "drone" / q["filename"]
        t0, c0 = time.perf_counter(), time.process_time()
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"cannot read {path}")
        ctx = QueryContext(bgr, embedder, frozen, verifier)
        setup_s = time.perf_counter() - t0
        tlat, tlon = float(q["lat"]), float(q["lon"])

        pending = []
        for r, k in wanted:
            if args.zero_offset:
                e, n = 0.0, 0.0
            elif args.control_offset_m is not None:
                e, n = float(args.control_offset_m), 0.0
            else:
                e, n = offset_for(seed, region, q["filename"], r, k)
            clat, clon = shift(tlat, tlon, e, n)
            res = localize_windowed(ctx, index, clat, clon, float(r), frozen, verifier)
            dist_c = geodesic_m(tlon, tlat, clon, clat)
            row = {
                "region": region, "group_7": g7[region], "filename": q["filename"], "num": q["num"],
                "variant": variant, "radius_m": r, "offset_k": k,
                "offset_east_m": round(e, 3), "offset_north_m": round(n, 3), "offset_dist_m": round(float(np.hypot(e, n)), 3),
                "center_lat": round(clat, 8), "center_lon": round(clon, 8),
                "truth_dist_to_center_m": round(dist_c, 3), "truth_within_radius": int(dist_c <= r),
                **res,
                "config_path": cfg_rel, "config_sha256": cfg_sha, "prereg_sha256": prereg_sha,
            }
            if res["pred_lat"] != "":
                err = geodesic_m(tlon, tlat, float(res["pred_lon"]), float(res["pred_lat"]))
                row["geodesic_error_m"] = round(err, 3)
            else:
                row["geodesic_error_m"] = ""
            acc = is_accepted(rule, row)
            row["accept_rule"] = rule or ""
            row["accepted"] = "" if acc is None else int(acc)
            if acc is None:
                row["outcome_primary"] = ""
            elif not acc:
                row["outcome_primary"] = "rejected"
            else:
                row["outcome_primary"] = "accepted_correct" if float(row["geodesic_error_m"]) <= tol else "accepted_incorrect"
            pending.append(row)

        cpu_s = time.process_time() - c0
        for row in pending:
            row["frame_setup_s"] = round(setup_s, 3)
            row["frame_cpu_s"] = round(cpu_s, 3)
            writer.writerow({k: row.get(k, "") for k in FIELDS})
            f_out.flush()
            new_rows += 1
        print(f"  {q['filename']} attempts={len(pending)} wall={time.perf_counter() - t0:.1f}s cpu={cpu_s:.1f}s", flush=True)

    f_out.close()
    print(f"done: {new_rows} new rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
