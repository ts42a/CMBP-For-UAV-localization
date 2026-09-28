#!/usr/bin/env python3
"""SUPERSEDED 2026-09-28 (deviation D1): selected `dinov2_c2f` over AKAZE-only, but
`dinov2_c2f` itself failed the Stage 4 zero-offset check and was replaced by the
DINOv2 patch verifier (configs/matcher_stage4_frozen.yaml). Kept for provenance of
the Stage 3B candidate comparison only; do not run this for label generation.

Stage 3B matcher recovery: tune on frozen tune-split, gate on untouched val-split.

Candidates (max 2):
  A) dinov2_global — frozen DINOv2 ViT-S/14 cosine retrieval over full tile DB
  B) dinov2_c2f    — same + AKAZE/RANSAC fine verify on top-k

CSV lat/lon never used for candidate selection. Stage 3A baseline preserved.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))

from dino_embedder import DinoGlobalEmbedder, load_embedding_cache  # noqa: E402
from fine_verify import akaze_rerank  # noqa: E402
from geo_utils import geodesic_m  # noqa: E402
from project_paths import (  # noqa: E402
    CACHE_EMB,
    DATASET_ROOT,
    MATCHER_PROTOCOL,
    SAT_META,
)
from query_select import load_manifest_region  # noqa: E402
from retrieval import top_k_similar  # noqa: E402
from tile_index import (  # noqa: E402
    TileSpec,
    build_tiles,
    georef_from_metadata,
    load_sat_memmap,
    read_tile_rgb,
)

SPLITS = ROOT / "configs" / "stage3b_splits.yaml"
OUT_CFG_A = ROOT / "configs" / "matcher_stage3b_dinov2_global.yaml"
OUT_CFG_B = ROOT / "configs" / "matcher_stage3b_dinov2_c2f.yaml"
OUT_FROZEN = ROOT / "configs" / "matcher_stage3b_frozen.yaml"
REPORTS = ROOT / "reports"
FIGS = ROOT / "figures" / "stage3b"
THRESHOLDS = (100, 500)


def sat_row(meta_rows, region: str) -> dict:
    for r in meta_rows:
        p = r["satellite_path"].replace("/", "\\")
        if f"\\{region}\\" in p and "satellite09_" not in r["filename"]:
            return r
    raise FileNotFoundError(region)


def lonlat_to_pixel(lat, lon, west, north, sx, sy):
    return (north - lat) / sy, (lon - west) / sx


def tiles_covering(tiles, row: float, col: float) -> set[int]:
    return {t.tile_id for t in tiles if t.row0 <= row < t.row1 and t.col0 <= col < t.col1}


def load_splits() -> dict:
    if not SPLITS.is_file():
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "localization" / "freeze_stage3b_splits.py")])
    return yaml.safe_load(SPLITS.read_text(encoding="utf-8"))


def rows_from_split(manifest: list[dict], entries: list[dict]) -> list[dict]:
    by_fn = {r["filename"]: r for r in manifest}
    out = []
    for e in entries:
        fn = e["filename"]
        if fn not in by_fn:
            raise KeyError(fn)
        out.append(by_fn[fn])
    return out


def query_embeddings(embedder: DinoGlobalEmbedder, bgr: np.ndarray, rotations: tuple[int, ...]) -> np.ndarray:
    """One L2-normalized embedding per rotation (no CSV heading)."""
    vecs = []
    for rot in rotations:
        if rot == 0:
            im = bgr
        else:
            M = cv2.getRotationMatrix2D((bgr.shape[1] / 2, bgr.shape[0] / 2), rot, 1.0)
            im = cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]), borderMode=cv2.BORDER_REFLECT)
        v = embedder.embed_rgb_arrays([cv2.cvtColor(im, cv2.COLOR_BGR2RGB)])[0]
        vecs.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(vecs, axis=0)


def top_k_multi_rot(embs: np.ndarray, qvecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Max cosine over query rotations, then top-k."""
    scores = np.max(embs @ qvecs.T, axis=1)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


def recall_rank(tiles, tile_ids_arr, idx, gt_lat, gt_lon, georef, hit_m: float = 150.0) -> int | None:
    west, north, sx, sy = georef
    gr, gc = lonlat_to_pixel(gt_lat, gt_lon, west, north, sx, sy)
    covering = tiles_covering(tiles, gr, gc)
    for rnk, i in enumerate(idx):
        tid = int(tile_ids_arr[int(i)]) if tile_ids_arr is not None else tiles[int(i)].tile_id
        t = tiles[int(i)]
        if tid in covering or t.tile_id in covering:
            return rnk + 1
        if geodesic_m(gt_lon, gt_lat, t.center_lon, t.center_lat) <= hit_m:
            return rnk + 1
    return None


def run_one(
    bgr,
    q,
    embedder,
    embs,
    tile_ids,
    tiles,
    sat,
    georef,
    mode: str,
    top_k: int,
    rotations: tuple[int, ...],
    fine_params: dict,
) -> dict:
    t0 = time.perf_counter()
    gt_lat, gt_lon = float(q["lat"]), float(q["lon"])
    qvecs = query_embeddings(embedder, bgr, rotations)
    idx, scores = top_k_multi_rot(embs, qvecs, top_k)
    rank = recall_rank(tiles, tile_ids, idx, gt_lat, gt_lon, georef)
    inliers = 0
    used_h = False
    if mode == "dinov2_c2f":
        fv = akaze_rerank(bgr, sat, tiles, [int(i) for i in idx], georef, fine_params)
        plat, plon = fv["pred_lat"], fv["pred_lon"]
        ref_id = fv["tile_id"]
        inliers = int(fv["inliers"])
        used_h = bool(fv["used_homography"])
        reason = "dino+akaze"
    else:
        t = tiles[int(idx[0])]
        plat, plon = t.center_lat, t.center_lon
        ref_id = t.tile_id
        reason = "dino_top1"
    err = geodesic_m(gt_lon, gt_lat, plon, plat)
    return {
        "query_id": q["num"],
        "region": "03",
        "filename": q["filename"],
        "gt_lat": gt_lat,
        "gt_lon": gt_lon,
        "pred_lat": plat,
        "pred_lon": plon,
        "predicted_tile_id": ref_id,
        "top_score": round(float(scores[0]), 5),
        "score_margin": round(float(scores[0] - scores[1]), 5) if len(scores) > 1 else 0.0,
        "geodesic_error_m": round(err, 3),
        "success_100m": int(err <= 100),
        "success_500m": int(err <= 500),
        "gt_in_topk_rank": rank if rank is not None else "",
        "recall_at_k": int(rank is not None),
        "fine_inliers": inliers,
        "used_homography": int(used_h),
        "runtime_s": round(time.perf_counter() - t0, 3),
        "matcher_id": mode,
        "search_used_gt_position": False,
        "selection_reason": reason,
        "failure_reason": "",
    }


def summarize(rows: list[dict], mode: str) -> dict:
    errs = [float(r["geodesic_error_m"]) for r in rows]
    n = len(rows)
    s100 = sum(int(r["success_100m"]) for r in rows)
    s500 = sum(int(r["success_500m"]) for r in rows)
    recalls = [int(r["recall_at_k"]) for r in rows]
    return {
        "mode": mode,
        "n": n,
        "success_100m": s100,
        "failure_100m": n - s100,
        "success_500m": s500,
        "success_rate_100m": s100 / n if n else 0.0,
        "success_rate_500m": s500 / n if n else 0.0,
        "median_error_m": float(np.median(errs)) if errs else None,
        "mean_error_m": float(np.mean(errs)) if errs else None,
        "topk_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_runtime_s": float(np.mean([float(r["runtime_s"]) for r in rows])) if rows else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def tune_on_dev(tune_q, embedder, embs, tile_ids, tiles, sat, georef) -> tuple[dict, list]:
    """Hyperparameter grid on TUNE split only."""
    grid = []
    fine_base = {
        "query_max_long_edge": 768,
        "ratio_test": 0.75,
        "min_inliers": 12,
        "ransac_reproj_threshold_px": 3.0,
    }
    for rotations in ((0,), (0, 90, 180, 270)):
        for top_k in (10, 20):
            for mode in ("dinov2_global", "dinov2_c2f"):
                rows = []
                for q in tune_q:
                    path = DATASET_ROOT / "03" / "drone" / q["filename"]
                    bgr = cv2.imread(str(path))
                    if bgr is None:
                        continue
                    rows.append(
                        run_one(
                            bgr, q, embedder, embs, tile_ids, tiles, sat, georef,
                            mode, top_k, rotations, fine_base,
                        )
                    )
                s = summarize(rows, mode)
                grid.append(
                    {
                        "mode": mode,
                        "top_k": top_k,
                        "rotations": list(rotations),
                        "success_100m": s["success_100m"],
                        "n": s["n"],
                        "median_error_m": s["median_error_m"],
                        "topk_recall": s["topk_recall"],
                    }
                )
                print(
                    f"  tune {mode} k={top_k} rot={rotations}: "
                    f"s100={s['success_100m']}/{s['n']} med={s['median_error_m']:.0f} "
                    f"R@{top_k}={s['topk_recall']:.2f}",
                    flush=True,
                )
    # Prefer useful mix (>=2 successes) then success count then recall then median
    def key(g):
        return (
            g["success_100m"] >= 2,
            g["success_100m"],
            g["topk_recall"],
            -(g["median_error_m"] or 1e9),
        )

    grid.sort(key=key, reverse=True)
    return grid[0], grid


def eval_set(queries, embedder, embs, tile_ids, tiles, sat, georef, mode, top_k, rotations, fine_params):
    rows = []
    for i, q in enumerate(queries):
        path = DATASET_ROOT / "03" / "drone" / q["filename"]
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rows.append(
            run_one(bgr, q, embedder, embs, tile_ids, tiles, sat, georef, mode, top_k, rotations, fine_params)
        )
        if (i + 1) % 10 == 0:
            print(f"    {mode} {i+1}/{len(queries)}", flush=True)
    return rows


def visual_verify(rows: list[dict], sat, georef, tag: str) -> list[dict]:
    """Human-inspectable panels + crude visual-credibility heuristic."""
    FIGS.mkdir(parents=True, exist_ok=True)
    west, north, sx, sy = georef
    judgments = []
    ranked = sorted(rows, key=lambda r: float(r["geodesic_error_m"]))
    # successes @100, near misses, worst failures
    picks = [r for r in ranked if int(r["success_100m"])][:5]
    picks += [r for r in ranked if not int(r["success_100m"]) and float(r["geodesic_error_m"]) < 800][:2]
    picks += ranked[-3:]
    seen = set()
    for r in picks:
        if r["filename"] in seen:
            continue
        seen.add(r["filename"])
        path = DATASET_ROOT / "03" / "drone" / r["filename"]
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        half = 256
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        lab = "succ100" if int(r["success_100m"]) else ("succ500" if int(r["success_500m"]) else "fail")
        axes[0].set_title(f"{r['filename']}\nerr={r['geodesic_error_m']:.1f}m ({lab})")
        axes[0].axis("off")
        for ax, lat, lon, title in (
            (axes[1], float(r["gt_lat"]), float(r["gt_lon"]), "sat @ GT"),
            (axes[2], float(r["pred_lat"]), float(r["pred_lon"]), "sat @ pred"),
        ):
            rr, cc = lonlat_to_pixel(lat, lon, west, north, sx, sy)
            r0, c0 = max(0, int(rr) - half), max(0, int(cc) - half)
            ax.imshow(np.asarray(sat[r0 : r0 + 2 * half, c0 : c0 + 2 * half, :3]))
            ax.set_title(title)
            ax.axis("off")
        out = FIGS / f"{tag}_{r['filename'].replace('.JPG', '')}_{lab}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)

        # Heuristic visual credibility: require GT neighbourhood in top-k AND
        # (homography used with enough inliers OR error very small with recall)
        credible = False
        if int(r["success_100m"]):
            rank_ok = r["gt_in_topk_rank"] != "" and int(r["gt_in_topk_rank"]) <= 20
            geo_ok = int(r.get("used_homography", 0)) == 1 and int(r.get("fine_inliers", 0)) >= 12
            # strict: positional hit alone is insufficient (Stage 3A lesson)
            credible = bool(rank_ok and (geo_ok or float(r["geodesic_error_m"]) <= 50))
        judgments.append(
            {
                "filename": r["filename"],
                "error_m": r["geodesic_error_m"],
                "success_100m": r["success_100m"],
                "gt_in_topk_rank": r["gt_in_topk_rank"],
                "fine_inliers": r.get("fine_inliers", 0),
                "used_homography": r.get("used_homography", 0),
                "visual_credible_100m": int(credible),
                "figure": str(out.relative_to(ROOT)).replace("\\", "/"),
                "notes": (
                    "credible iff @100m AND GT in top-k AND (homography+inliers>=12 OR err<=50m)"
                    if int(r["success_100m"])
                    else "failure_or_non100"
                ),
            }
        )
    return judgments


def load_baseline_for(filenames: list[str]) -> list[dict]:
    p = REPORTS / "stage3a_pilot_results.csv"
    by = {r["filename"]: r for r in csv.DictReader(p.open(encoding="utf-8", newline=""))}
    out = []
    for fn in filenames:
        b = by.get(fn)
        if not b:
            continue
        err = float(b["geodesic_error_m"])
        out.append(
            {
                "filename": fn,
                "geodesic_error_m": err,
                "success_100m": int(err <= 100),
                "success_500m": int(err <= 500),
                "runtime_s": float(b.get("runtime_s") or 0),
                "matcher_id": "pyramid_ncc_rot4",
                "recall_at_k": "",
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-diagnose", action="store_true")
    args = ap.parse_args()

    if not args.skip_diagnose:
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "localization" / "stage3b_diagnose.py")])

    splits = load_splits()
    print(f"Splits frozen_at={splits['frozen_at']} val={splits['n_val']} tune={splits['n_tune']}", flush=True)

    with MATCHER_PROTOCOL.open(encoding="utf-8") as f:
        proto = yaml.safe_load(f)

    cache_path = CACHE_EMB / "region03_dinov2_vits14_224.npz"
    if not cache_path.is_file():
        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts" / "extract_reference_embeddings.py"), "--region", "03"]
        )
    embs, tile_ids, cache_meta = load_embedding_cache(cache_path)

    with SAT_META.open(encoding="utf-8", newline="") as f:
        meta_rows = list(csv.DictReader(f))
    sat_meta = sat_row(meta_rows, "03")
    sat_path = Path(sat_meta["satellite_path"])
    if not sat_path.is_file():
        sat_path = DATASET_ROOT / "03" / sat_meta["filename"]
    west, north, sx, sy = georef_from_metadata(sat_meta)
    georef = (west, north, sx, sy)
    sat, (h, w) = load_sat_memmap(sat_path)
    spec = TileSpec(int(proto["satellite_tiling"]["tile_size_px"]), int(proto["satellite_tiling"]["stride_px"]))
    tiles = build_tiles(h, w, west, north, sx, sy, spec)
    assert len(tiles) == len(embs)

    manifest = load_manifest_region("03")
    tune_q = rows_from_split(manifest, splits["tune_queries"])
    val_q = rows_from_split(manifest, splits["val_queries"])
    compare_fns = [e["filename"] for e in splits["compare_stage3a_queries"]]
    compare_q = rows_from_split(manifest, [{"filename": f} for f in compare_fns])

    # Ensure no tune/val leakage
    assert not ({r["filename"] for r in tune_q} & {r["filename"] for r in val_q})

    embedder = DinoGlobalEmbedder(image_size=224)
    print("Tuning on tune-split only...", flush=True)
    best, grid = tune_on_dev(tune_q, embedder, embs, tile_ids, tiles, sat, georef)
    print("Selected on tune:", best, flush=True)

    rotations = tuple(best["rotations"])
    top_k = int(best["top_k"])
    fine_params = {
        "query_max_long_edge": 768,
        "ratio_test": 0.75,
        "min_inliers": 12,
        "ransac_reproj_threshold_px": 3.0,
    }

    # Evaluate BOTH candidates on VAL with frozen hyperparams from best *per-mode* on tune
    # Re-pick best hyperparams per mode from grid (still tune-only)
    by_mode_best = {}
    for mode in ("dinov2_global", "dinov2_c2f"):
        cands = [g for g in grid if g["mode"] == mode]
        cands.sort(
            key=lambda g: (g["success_100m"], g["topk_recall"], -(g["median_error_m"] or 1e9)),
            reverse=True,
        )
        by_mode_best[mode] = cands[0]

    val_rows = {}
    val_sum = {}
    judgments = {}
    for mode, hp in by_mode_best.items():
        print(f"VAL eval {mode} k={hp['top_k']} rot={hp['rotations']}...", flush=True)
        rows = eval_set(
            val_q, embedder, embs, tile_ids, tiles, sat, georef,
            mode, int(hp["top_k"]), tuple(hp["rotations"]), fine_params,
        )
        val_rows[mode] = rows
        val_sum[mode] = summarize(rows, mode)
        write_csv(REPORTS / f"stage3b_val_{mode}.csv", rows)
        judgments[mode] = visual_verify(rows, sat, georef, f"val_{mode}")
        write_csv(REPORTS / f"stage3b_val_{mode}_visual.csv", judgments[mode])

    # Compare on Stage 3A query list (both candidates + baseline)
    print("Compare on Stage 3A query set...", flush=True)
    compare_rows = {}
    compare_sum = {}
    for mode, hp in by_mode_best.items():
        rows = eval_set(
            compare_q, embedder, embs, tile_ids, tiles, sat, georef,
            mode, int(hp["top_k"]), tuple(hp["rotations"]), fine_params,
        )
        compare_rows[mode] = rows
        compare_sum[mode] = summarize(rows, mode)
        write_csv(REPORTS / f"stage3b_compare_{mode}.csv", rows)
        judgments[f"compare_{mode}"] = visual_verify(rows, sat, georef, f"cmp_{mode}")

    baseline = load_baseline_for(compare_fns)
    compare_sum["pyramid_ncc_rot4"] = summarize(
        [
            {
                "geodesic_error_m": b["geodesic_error_m"],
                "success_100m": b["success_100m"],
                "success_500m": b["success_500m"],
                "runtime_s": b["runtime_s"],
                "recall_at_k": 0,
            }
            for b in baseline
        ],
        "pyramid_ncc_rot4",
    )
    write_csv(REPORTS / "stage3b_compare_pyramid_ncc_rot4.csv", baseline)

    # Side-by-side comparison CSV
    by_base = {b["filename"]: b for b in baseline}
    joint = []
    for fn in compare_fns:
        row = {"filename": fn}
        b = by_base.get(fn, {})
        row["baseline_error_m"] = b.get("geodesic_error_m", "")
        row["baseline_success_100m"] = b.get("success_100m", "")
        for mode in ("dinov2_global", "dinov2_c2f"):
            m = next((r for r in compare_rows[mode] if r["filename"] == fn), None)
            if m:
                row[f"{mode}_error_m"] = m["geodesic_error_m"]
                row[f"{mode}_success_100m"] = m["success_100m"]
                row[f"{mode}_recall_at_k"] = m["recall_at_k"]
                row[f"{mode}_runtime_s"] = m["runtime_s"]
        joint.append(row)
    write_csv(REPORTS / "stage3b_comparison_per_query.csv", joint)

    # Candidate configs
    meta = embedder.meta()
    for mode, path in (("dinov2_global", OUT_CFG_A), ("dinov2_c2f", OUT_CFG_B)):
        hp = by_mode_best[mode]
        cfg = {
            "schema_version": "stage3b_candidate_v1",
            "candidate_id": mode,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "model": {k: str(v) for k, v in meta.__dict__.items()},
            "weights_provenance": {
                "source": "torch.hub facebookresearch/dinov2 dinov2_vits14 pretrained=True",
                "note": "Proposal DINOv3 unavailable; DINOv2 ViT-S/14 documented substitute.",
            },
            "preprocessing": {
                "image_size": 224,
                "imagenet_normalize": True,
                "query_rotations_deg": hp["rotations"],
                "query_rotation_aggregation": "max_cosine_over_rotations",
                "csv_heading_used": False,
            },
            "retrieval": {"metric": "cosine", "top_k": int(hp["top_k"]), "database": "full_region_tile_index"},
            "fine_match": {
                "enabled": mode == "dinov2_c2f",
                "method": "akaze_ransac_homography",
                **{k: (float(v) if isinstance(v, (float,)) else int(v) if isinstance(v, (int,)) else v) for k, v in fine_params.items()},
            },
            "resources": {
                "device": str(meta.device),
                "torch": str(meta.torch_version),
                "ram_target_gb": 16,
            },
            "tune_only": {
                "split_file": str(SPLITS),
                "selected_on_tune": {
                    "mode": hp["mode"],
                    "top_k": int(hp["top_k"]),
                    "rotations": [int(x) for x in hp["rotations"]],
                    "success_100m": int(hp["success_100m"]),
                    "n": int(hp["n"]),
                    "median_error_m": float(hp["median_error_m"]),
                    "topk_recall": float(hp["topk_recall"]),
                },
            },
            "tiling": {
                "tile_size_px": int(proto["satellite_tiling"]["tile_size_px"]),
                "stride_px": int(proto["satellite_tiling"]["stride_px"]),
            },
        }
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Gate on VAL: need useful mix of *visually credible* successes and failures @100m
    gate_mode = None
    gate_detail = {}
    for mode in ("dinov2_c2f", "dinov2_global"):
        s = val_sum[mode]
        vis = judgments[mode]
        n_cred = sum(int(v["visual_credible_100m"]) for v in vis)
        n_pos = s["success_100m"]
        n_fail = s["failure_100m"]
        mixed_pos = n_pos >= 2 and n_fail >= 2
        mixed_cred = n_cred >= 2 and n_fail >= 2
        gate_detail[mode] = {
            "val_success_100m": n_pos,
            "val_failure_100m": n_fail,
            "val_visual_credible_100m": n_cred,
            "mixed_positional": mixed_pos,
            "mixed_visual_credible": mixed_cred,
            "median_error_m": s["median_error_m"],
            "topk_recall": s["topk_recall"],
        }
        if mixed_cred and gate_mode is None:
            gate_mode = mode

    ready = gate_mode is not None
    decision = "READY" if ready else "NOT READY"

    if ready:
        # Freeze winner
        winner_cfg = yaml.safe_load(OUT_CFG_B.read_text(encoding="utf-8") if gate_mode == "dinov2_c2f" else OUT_CFG_A.read_text(encoding="utf-8"))
        winner_cfg["matcher_selected"] = gate_mode
        winner_cfg["gate"] = "READY"
        winner_cfg["gate_basis"] = "untouched_val_visual_credible_mix_100m"
        OUT_FROZEN.write_text(yaml.safe_dump(winner_cfg, sort_keys=False), encoding="utf-8")
    else:
        OUT_FROZEN.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "stage3b_frozen_v1",
                    "matcher_selected": None,
                    "gate": "NOT READY",
                    "frozen_at": datetime.now(timezone.utc).isoformat(),
                    "reason": "No candidate produced a useful mix of visually credible @100m successes on untouched val.",
                    "val_gate_detail": gate_detail,
                    "baseline_preserved": "pyramid_ncc_rot4",
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    summary = {
        "decision": decision,
        "gate_mode": gate_mode,
        "val_summaries": val_sum,
        "val_gate_detail": gate_detail,
        "compare_summaries": compare_sum,
        "tune_best_per_mode": by_mode_best,
        "tune_grid": grid,
        "splits": {
            "path": str(SPLITS),
            "frozen_at": splits["frozen_at"],
            "n_val": splits["n_val"],
            "n_tune": splits["n_tune"],
        },
        "embedding_cache": str(cache_path),
        "cache_meta": cache_meta.get("meta", {}),
    }
    (REPORTS / "stage3b_recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Decision markdown
    alt = ""
    if not ready:
        alt = """
## Recommended alternative (research-task revision)

None of the Stage 3B candidates met the freeze rule on the **untouched validation** subset.

Concrete alternatives (pick one before any full-dataset labeling):

1. **Cross-view VPR / UAV-specific retrieval** — use a model pretrained or fine-tuned for aerial↔satellite
   (e.g. published UAV-VisLoc / GLVL baselines from the dataset paper), still with full-map search and
   geographic holdouts. Do not invent labels from NCC.
2. **Restricted-pose protocol (separate study)** — if the scientific goal allows, evaluate under a
   *declared* prior on heading from an independent IMU (not CSV Phi until validated) and/or altitude;
   this changes the task and must be pre-registered.
3. **Map-matchability segmentation labels (RSPNet-style)** — revise the research task away from
   real matcher success prediction toward map distinctiveness segmentation; this no longer fills the
   four-criterion gap as stated in Stage 1.
4. **Do not** widen the 100 m threshold, train the localizability predictor, or generate 6,774 labels
   with the current matchers.
"""

    decision_md = f"""# Stage 3B — Decision

**Decision: {decision}** for full real-outcome label generation.

## Evidence (untouched validation subset, region 03)

Splits frozen in `configs/stage3b_splits.yaml` at `{splits['frozen_at']}`  
**before** hyperparameter tuning (`n_val={splits['n_val']}`, `n_tune={splits['n_tune']}`).

| Candidate | Val @100 m succ/fail | Visual-credible @100 m | Top-k recall | Median err (m) |
|---|---:|---:|---:|---:|
| dinov2_global | {gate_detail['dinov2_global']['val_success_100m']}/{gate_detail['dinov2_global']['val_failure_100m']} | {gate_detail['dinov2_global']['val_visual_credible_100m']} | {gate_detail['dinov2_global']['topk_recall']:.2f} | {gate_detail['dinov2_global']['median_error_m']:.0f} |
| dinov2_c2f | {gate_detail['dinov2_c2f']['val_success_100m']}/{gate_detail['dinov2_c2f']['val_failure_100m']} | {gate_detail['dinov2_c2f']['val_visual_credible_100m']} | {gate_detail['dinov2_c2f']['topk_recall']:.2f} | {gate_detail['dinov2_c2f']['median_error_m']:.0f} |

Freeze rule: useful mix of **visually credible** successes **and** failures at **100 m** on val  
(positional-only hits are insufficient — Stage 3A @500 m lesson).

Selected for freeze: `{gate_mode}`  
Config: `configs/matcher_stage3b_frozen.yaml`

## Stage 3A baseline (preserved)

`pyramid_ncc_rot4` on the Stage 3A 100-query set: **0/100** @100 m, median ~3.3 km  
(see `reports/stage3a_pilot_summary.md`). Not overwritten.

## Comparison on Stage 3A queries (not used for matcher choice)

| Mode | n | @100 m | @500 m | Median err (m) | Top-k recall | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| pyramid_ncc_rot4 | {compare_sum['pyramid_ncc_rot4']['n']} | {compare_sum['pyramid_ncc_rot4']['success_100m']} | {compare_sum['pyramid_ncc_rot4']['success_500m']} | {compare_sum['pyramid_ncc_rot4']['median_error_m']:.0f} | n/a | {compare_sum['pyramid_ncc_rot4']['mean_runtime_s']:.2f} |
| dinov2_global | {compare_sum['dinov2_global']['n']} | {compare_sum['dinov2_global']['success_100m']} | {compare_sum['dinov2_global']['success_500m']} | {compare_sum['dinov2_global']['median_error_m']:.0f} | {compare_sum['dinov2_global']['topk_recall']:.2f} | {compare_sum['dinov2_global']['mean_runtime_s']:.2f} |
| dinov2_c2f | {compare_sum['dinov2_c2f']['n']} | {compare_sum['dinov2_c2f']['success_100m']} | {compare_sum['dinov2_c2f']['success_500m']} | {compare_sum['dinov2_c2f']['median_error_m']:.0f} | {compare_sum['dinov2_c2f']['topk_recall']:.2f} | {compare_sum['dinov2_c2f']['mean_runtime_s']:.2f} |

Per-query: `reports/stage3b_comparison_per_query.csv`  
Figures: `figures/stage3b/`

## What this decision does **not** claim

- No held-out **region** performance (val is still region 03)
- No calibration / scheduling benefit
- No localizability predictor training
- No full 6,774 label generation unless decision is READY
{alt}
## Reproduction

```bash
py -3 F:\\CMBL\\scripts\\localization\\freeze_stage3b_splits.py
py -3 F:\\CMBL\\scripts\\run_stage3b_recovery.py
```
"""
    (REPORTS / "stage3b_decision.md").write_text(decision_md, encoding="utf-8")
    print(json.dumps({"decision": decision, "gate_mode": gate_mode, "val_gate_detail": gate_detail}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
