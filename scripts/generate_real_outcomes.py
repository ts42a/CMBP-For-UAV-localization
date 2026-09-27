#!/usr/bin/env python3
"""Stage 3C — resumable full real-outcome labeling with the frozen dinov2_c2f matcher.

Usage:
  py -3 scripts/localization/stage3c_gate.py
  py -3 scripts/generate_real_outcomes.py --smoke
  py -3 scripts/generate_real_outcomes.py --extract-embeddings
  py -3 scripts/generate_real_outcomes.py --run --resume
  py -3 scripts/generate_real_outcomes.py --audit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from frozen_matcher import localize_query, load_frozen  # noqa: E402
from geo_utils import geodesic_m  # noqa: E402
from project_paths import CACHE_EMB, DATASET_ROOT, MANIFEST  # noqa: E402
from query_select import load_manifest_region  # noqa: E402
from region_map import load_region  # noqa: E402
from tile_index import TileSpec  # noqa: E402

GATE = ROOT / "reports" / "stage3c_preflight_gate.json"
OUT_DIR = ROOT / "results" / "stage3c"
MANIFEST_OUT = OUT_DIR / "run_manifest.json"
COMBINED = OUT_DIR / "outcomes.csv"
FIELDS = [
    "query_id",
    "region",
    "filename",
    "pred_lat",
    "pred_lon",
    "predicted_tile_id",
    "predicted_raster_id",
    "top_score",
    "score_margin",
    "top_k",
    "fine_inliers",
    "used_homography",
    "geo_verify_status",
    "matcher_failure_reason",
    "localization_failure_reason",
    "scoring_lat",
    "scoring_lon",
    "geodesic_error_m",
    "success_100m",
    "success_500m",
    "runtime_s",
    "matcher_id",
    "search_used_gt_position",
]
REGIONS = [f"{i:02d}" for i in range(1, 12)]
SMOKE_REGIONS = ["01", "03", "05", "07", "09"]


def require_gate() -> dict:
    if not GATE.is_file():
        raise SystemExit("Run scripts/localization/stage3c_gate.py first.")
    g = json.loads(GATE.read_text(encoding="utf-8"))
    if g.get("gate") != "PASS":
        raise SystemExit(f"NOT READY: preflight gate failed: {g.get('causes')}")
    return g


def load_all_queries() -> list[dict]:
    rows = []
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def region_outcome_path(region: str) -> Path:
    return OUT_DIR / f"outcomes_region{region}.csv"


def done_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {r["filename"] for r in csv.DictReader(f) if r.get("filename")}


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def score_row(pred: dict, q: dict) -> dict:
    slat, slon = float(q["lat"]), float(q["lon"])
    out = {
        "query_id": q["num"],
        "region": q["region_folder"],
        "filename": q["filename"],
        **pred,
        "scoring_lat": slat,
        "scoring_lon": slon,
        "localization_failure_reason": "",
        "geodesic_error_m": "",
        "success_100m": "",
        "success_500m": "",
    }
    if pred.get("matcher_failure_reason"):
        out["localization_failure_reason"] = "matcher_failed_no_prediction"
        return out
    plat, plon = float(pred["pred_lat"]), float(pred["pred_lon"])
    err = geodesic_m(slon, slat, plon, plat)
    out["geodesic_error_m"] = round(err, 3)
    out["success_100m"] = int(err <= 100)
    out["success_500m"] = int(err <= 500)
    if err > 100:
        out["localization_failure_reason"] = "error_gt_100m"
    return out


def rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def freeze_manifest(frozen: dict, extra: dict) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git = "unknown"
    cfg_hash = hashlib.sha256((ROOT / "configs" / "matcher_stage3b_frozen.yaml").read_bytes()).hexdigest()
    queries = load_all_queries()
    counts = {}
    for q in queries:
        counts[q["region_folder"]] = counts.get(q["region_folder"], 0) + 1
    payload = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git,
        "config_path": "configs/matcher_stage3b_frozen.yaml",
        "config_sha256": cfg_hash,
        "matcher_id": frozen["matcher_selected"],
        "model": frozen["model"],
        "weights_provenance": frozen["weights_provenance"],
        "retrieval": frozen["retrieval"],
        "fine_match": frozen["fine_match"],
        "preprocessing": frozen["preprocessing"],
        "success_thresholds_m": [100, 500],
        "primary_threshold_m": 100,
        "dataset": {
            "identity": "GitHub IntelliSensing/UAV-VisLoc extract",
            "root": str(DATASET_ROOT),
            "n_queries_expected": 6774,
            "n_queries_manifest": len(queries),
            "per_region": counts,
            "note_6742_vs_6774": "Use 6774; README 6742 is documentation drift (Stage 1-2).",
        },
        "search_uses_gt_position": False,
        "csv_coordinates_role": "scoring_only",
        "baseline_preserved": "pyramid_ncc_rot4 Stage 3A",
        "no_tuning_on_held_out": True,
        **extra,
    }
    MANIFEST_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_region(
    region: str,
    queries: list[dict],
    frozen: dict,
    embedder: DinoGlobalEmbedder,
    resume: bool,
    limit: int | None = None,
) -> list[dict]:
    spec = TileSpec(
        int(frozen["retrieval"]["tiling"]["tile_size_px"]),
        int(frozen["retrieval"]["tiling"]["stride_px"]),
    )
    rasters, tiles = load_region(region, spec)
    cache = CACHE_EMB / f"region{region}_dinov2_vits14_{int(frozen['model']['image_size'])}.npz"
    if not cache.is_file():
        raise FileNotFoundError(f"Missing embeddings {cache}; run --extract-embeddings")
    embs, tile_ids, _ = load_embedding_cache(cache)
    if len(embs) != len(tiles):
        raise RuntimeError(f"Region {region}: embeddings {len(embs)} != tiles {len(tiles)}")

    out_path = region_outcome_path(region)
    done = done_keys(out_path) if resume else set()
    if not resume and out_path.exists():
        out_path.unlink()
        done = set()

    written = []
    n_run = 0
    for q in queries:
        if q["filename"] in done:
            continue
        path = Path(q["image_path"]) if q.get("image_path") else DATASET_ROOT / region / "drone" / q["filename"]
        bgr = cv2.imread(str(path))
        pred = localize_query(bgr, embedder, embs, tiles, rasters, frozen)
        row = score_row(pred, q)
        append_row(out_path, row)
        written.append(row)
        n_run += 1
        if n_run % 10 == 0:
            print(f"  {region} {n_run} new / rss={rss_mb()}", flush=True)
        if limit is not None and n_run >= limit:
            break
    return written


def cmd_smoke(frozen: dict) -> int:
    require_gate()
    print("Smoke test: embeddings for smoke regions...", flush=True)
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "extract_all_reference_embeddings.py"),
            "--regions",
            ",".join(SMOKE_REGIONS),
        ]
    )
    embedder = DinoGlobalEmbedder(image_size=int(frozen["model"]["image_size"]))
    smoke_rows = []
    for region in SMOKE_REGIONS:
        qs = load_manifest_region(region)
        pick = [qs[0], qs[len(qs) // 2], qs[-1]]
        print(f"Smoke region {region} n_pick={len(pick)}", flush=True)
        t0 = time.perf_counter()
        rows = run_region(region, pick, frozen, embedder, resume=False, limit=3)
        dt = time.perf_counter() - t0
        spec = TileSpec(1024, 512)
        rasters, tiles = load_region(region, spec)
        print(
            f"  tiles={len(tiles)} rasters={[r.filename for r in rasters]} "
            f"runtime={dt:.1f}s rss={rss_mb()} rows={len(rows)}",
            flush=True,
        )
        for r in rows:
            smoke_rows.append(r)
            print(
                f"    {r['filename']} pred=({r['pred_lat']},{r['pred_lon']}) "
                f"err={r['geodesic_error_m']} geo={r['geo_verify_status']} "
                f"mf={r['matcher_failure_reason']}",
                flush=True,
            )
    smoke_path = OUT_DIR / "smoke_outcomes.csv"
    if smoke_path.exists():
        smoke_path.unlink()
    for r in smoke_rows:
        append_row(smoke_path, r)
    (OUT_DIR / "smoke_report.json").write_text(
        json.dumps(
            {
                "n": len(smoke_rows),
                "regions": SMOKE_REGIONS,
                "schema_ok": all(k in smoke_rows[0] for k in FIELDS) if smoke_rows else False,
                "any_matcher_failure": any(r["matcher_failure_reason"] for r in smoke_rows),
                "rss_mb": rss_mb(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # do not keep smoke rows in region CSVs (would pollute resume)
    for region in SMOKE_REGIONS:
        p = region_outcome_path(region)
        if p.exists():
            p.unlink()
    print("Smoke OK", flush=True)
    return 0


def cmd_run(frozen: dict, resume: bool, regions: list[str]) -> int:
    require_gate()
    extra = {"phase": "full_run", "resume": resume, "rss_mb_start": rss_mb()}
    freeze_manifest(frozen, extra)
    embedder = DinoGlobalEmbedder(image_size=int(frozen["model"]["image_size"]))
    for region in regions:
        qs = load_manifest_region(region)
        print(f"RUN region {region} n={len(qs)} resume={resume}", flush=True)
        run_region(region, qs, frozen, embedder, resume=resume)
    combine_outcomes()
    return 0


def combine_outcomes() -> list[dict]:
    all_rows = []
    seen = set()
    dups = 0
    for region in REGIONS:
        p = region_outcome_path(region)
        if not p.is_file():
            continue
        with p.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                key = (r["region"], r["filename"])
                if key in seen:
                    dups += 1
                    continue
                seen.add(key)
                all_rows.append(r)
    COMBINED.parent.mkdir(parents=True, exist_ok=True)
    with COMBINED.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    mf = [r for r in all_rows if r.get("matcher_failure_reason")]
    with (OUT_DIR / "matcher_failures.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(mf)
    (OUT_DIR / "combine_audit.json").write_text(
        json.dumps({"n": len(all_rows), "duplicates_skipped": dups, "matcher_failures": len(mf)}, indent=2),
        encoding="utf-8",
    )
    return all_rows


def cmd_audit() -> int:
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "localization" / "stage3c_audit.py")])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extract-embeddings", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--regions", default="all")
    args = ap.parse_args()

    frozen = load_frozen()
    regions = REGIONS if args.regions == "all" else [x.strip() for x in args.regions.split(",")]

    if args.smoke:
        return cmd_smoke(frozen)
    if args.extract_embeddings:
        require_gate()
        return subprocess.call(
            [sys.executable, str(ROOT / "scripts" / "extract_all_reference_embeddings.py"), "--regions", args.regions]
        )
    if args.run:
        return cmd_run(frozen, args.resume, regions)
    if args.audit:
        return cmd_audit()

    print("Specify --smoke, --extract-embeddings, --run, or --audit")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
