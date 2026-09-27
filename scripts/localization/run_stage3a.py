#!/usr/bin/env python3
"""Stage 3A — dev matcher selection + ~100-query pilot (region 03)."""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(r"F:\CMBL")
sys.path.insert(0, str(ROOT / "scripts" / "localization"))

from geo_utils import geodesic_m, pixel_to_lonlat  # noqa: E402
from matchers import _prep_query, coarse_ncc_scores, precompute_tile_coarse  # noqa: E402
from pyramid_ncc import pyramid_ncc_predict  # noqa: E402
from tile_index import TileSpec, build_tiles, georef_from_metadata, load_sat_memmap  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "matcher_protocol.yaml"
REPORTS = ROOT / "reports"
FIGS = ROOT / "figures" / "stage3a"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest(region: str) -> list[dict]:
    rows = []
    with (ROOT / "datasets" / "manifests" / "manifest.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["region_folder"] == region:
                rows.append(r)
    return sorted(rows, key=lambda x: int(x["num"]))


def select_queries(manifest: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified deciles then even fill."""
    by_num = {int(r["num"]): r for r in manifest}
    nmax = max(by_num)
    deciles = {max(1, min(nmax, int(round((d / 10) * nmax)))) for d in range(1, 11)}
    chosen = [by_num[k] for k in sorted(deciles) if k in by_num]
    seen = {int(r["num"]) for r in chosen}
    if len(chosen) < n:
        step = max(1, nmax // (n - len(chosen) + 1))
        for num in range(1, nmax + 1, step):
            if num not in seen:
                chosen.append(by_num[num])
                seen.add(num)
            if len(chosen) >= n:
                break
    rng = np.random.default_rng(seed)
    if len(chosen) > n:
        idx = rng.choice(len(chosen), size=n, replace=False)
        chosen = [chosen[i] for i in sorted(idx)]
    return sorted(chosen[:n], key=lambda r: int(r["num"]))


def sat_row_for_region(meta_rows: list[dict], region: str) -> dict:
    for r in meta_rows:
        if f"\\{region}\\" in r["satellite_path"].replace("/", "\\") or f"/{region}/" in r["satellite_path"]:
            if "satellite09_" not in r["filename"]:
                return r
    raise FileNotFoundError(f"No canonical satellite for region {region}")


def run_one(
    matcher_id: str,
    query_path: Path,
    sat,
    tiles,
    georef,
    cfg: dict,
) -> dict:
    west, north, sx, sy = georef
    params = cfg["matcher_params"].get(matcher_id, {})
    max_long = int(cfg["query_preprocessing"]["max_long_edge_px"])
    t0 = time.perf_counter()
    bgr = cv2.imread(str(query_path))
    if bgr is None:
        return {
            "failure_reason": "read_error",
            "runtime_s": time.perf_counter() - t0,
        }
    fn = MATCHERS[matcher_id]
    res = fn(bgr, sat, tiles, params, max_long)
    runtime = time.perf_counter() - t0

    pred_lat, pred_lon = res.pred_lat, res.pred_lon
    if matcher_id == "akaze_coarse_to_fine" and res.failure_reason is None:
        # homography path: re-run centre mapping (matchers return tile centre as fallback)
        res2 = match_akaze_c2f(bgr, sat, tiles, params, max_long)
        t = tiles[res2.tile_id]
        # If homography succeeded, fr/fc encoded only in successful branch — compute from tile + offset
        pred_lat, pred_lon = t.center_lat, t.center_lon
        res = res2

    # For akaze with homography success, refine using stored tile and re-match once for pixel
    if matcher_id == "akaze_coarse_to_fine":
        pred_lat, pred_lon, tile_id, score, margin, fail = _akaze_predict_lonlat(
            bgr, sat, tiles, georef, params, max_long
        )
    else:
        tile_id, score, margin, fail = res.tile_id, res.top_score, res.score_margin, res.failure_reason
        t = tiles[tile_id]
        pred_lat, pred_lon = t.center_lat, t.center_lon

    return {
        "predicted_tile_id": tile_id,
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "top_score": score,
        "score_margin": margin,
        "failure_reason": fail,
        "runtime_s": runtime,
    }


def _akaze_predict_lonlat(bgr, sat, tiles, georef, params, max_long, tile_coarse=None):
    from tile_index import read_tile_rgb  # noqa: E402

    west, north, sx, sy = georef
    _, qg = _prep_query(bgr, max_long)
    cpx = params.get("coarse_patch_px", 64)
    scores = coarse_ncc_scores(qg, sat, tiles, cpx, tile_coarse)
    top_k = int(params.get("coarse_top_k", 15))
    order = np.argsort(-scores)[:top_k]
    akaze = cv2.AKAZE_create()
    kp_q, des_q = akaze.detectAndCompute(qg, None)
    if des_q is None:
        t = tiles[int(order[0])]
        return t.center_lat, t.center_lon, t.tile_id, float(scores[order[0]]), 0.0, "no_query_features"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    ratio = float(params.get("ratio_test", 0.75))
    min_inl = int(params.get("min_inliers_for_homography", 12))
    ransac_th = float(params.get("ransac_reproj_threshold_px", 3.0))
    best_inl = -1
    second_inl = -1
    best = None
    fail = "insufficient_inliers"

    for idx in order:
        t = tiles[int(idx)]
        patch = read_tile_rgb(sat, t)
        pg = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        kp_t, des_t = akaze.detectAndCompute(pg, None)
        if des_t is None:
            continue
        pairs = bf.knnMatch(des_q, des_t, k=2)
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
        if len(good) < 8:
            continue
        src = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_th)
        inliers = int(mask.sum()) if mask is not None else 0
        if inliers > best_inl:
            second_inl = best_inl
            best_inl = inliers
            best = (t, H, inliers)
            if H is not None and inliers >= min_inl:
                fail = None

    if best is None:
        t = tiles[int(order[0])]
        return (
            t.center_lat,
            t.center_lon,
            t.tile_id,
            float(scores[order[0]]),
            0.0,
            "no_tile_features",
        )

    t, H, inliers = best
    qh, qw = qg.shape[:2]
    if fail is None and H is not None:
        pt = np.array([[[qw / 2, qh / 2]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)[0, 0]
        fr = t.row0 + mapped[1]
        fc = t.col0 + mapped[0]
        pred_lat, pred_lon = pixel_to_lonlat(fr, fc, west, north, sx, sy)
        margin = float(inliers - max(second_inl, 0))
        return pred_lat, pred_lon, t.tile_id, float(inliers), margin, None

    margin = float(inliers - max(second_inl, 0))
    return t.center_lat, t.center_lon, t.tile_id, float(inliers), margin, fail


def evaluate_matcher(
    matcher_id: str,
    queries: list[dict],
    sat,
    tiles,
    georef,
    cfg: dict,
    thresholds: list[float],
    tile_coarse,
) -> dict:
    errs = []
    succ = {t: 0 for t in thresholds}
    max_long = int(cfg["query_preprocessing"]["max_long_edge_px"])
    params = cfg["matcher_params"].get(matcher_id, {})
    for q in queries:
        path = ROOT / "datasets" / "raw" / "UAV_VisLoc_dataset" / "03" / "drone" / q["filename"]
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        if matcher_id == "pyramid_ncc_rot4":
            p = cfg["matcher_params"]["pyramid_ncc_rot4"]
            sat_rgb = np.asarray(sat[:, :, :3])
            pred_lat, pred_lon, _, _, _ = pyramid_ncc_predict(
                bgr,
                sat_rgb,
                georef,
                int(p["max_long_edge_px"]),
                int(p["satellite_max_long_edge_px"]),
                tuple(int(x) for x in p["rotations_deg"]),
            )
        elif matcher_id == "akaze_coarse_to_fine":
            pred_lat, pred_lon, _, _, _, _ = _akaze_predict_lonlat(
                bgr, sat, tiles, georef, params, max_long, tile_coarse
            )
        else:
            _, qg = _prep_query(bgr, max_long)
            scores = coarse_ncc_scores(qg, sat, tiles, params.get("coarse_patch_px", 64), tile_coarse)
            t = tiles[int(np.argmax(scores))]
            pred_lat, pred_lon = t.center_lat, t.center_lon
        gt_lat, gt_lon = float(q["lat"]), float(q["lon"])
        err = geodesic_m(gt_lon, gt_lat, pred_lon, pred_lat)
        errs.append(err)
        for t in thresholds:
            if err <= t:
                succ[t] += 1
    n = len(errs)
    return {
        "n": n,
        "median_error_m": float(np.median(errs)) if errs else None,
        "success_rates": {str(t): succ[t] / n if n else 0 for t in thresholds},
    }


def write_pilot_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "query_id",
        "region",
        "filename",
        "gt_lat",
        "gt_lon",
        "pred_lat",
        "pred_lon",
        "predicted_tile_id",
        "top_score",
        "score_margin",
        "geodesic_error_m",
        "success_50m",
        "success_100m",
        "success_250m",
        "success_500m",
        "runtime_s",
        "failure_reason",
        "matcher_id",
        "search_used_gt_position",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def lonlat_to_pixel(lon: float, lat: float, georef: tuple) -> tuple[float, float]:
    west, north, sx, sy = georef
    col = (lon - west) / sx
    row = (north - lat) / sy
    return row, col


def make_figures(pilot_rows: list[dict], tiles, sat, cfg: dict, georef) -> None:
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    errs = [r["geodesic_error_m"] for r in pilot_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errs, bins=30, color="steelblue", edgecolor="white")
    ax.axvline(100, color="red", ls="--", label="100 m threshold")
    ax.set_xlabel("Geodesic error (m)")
    ax.set_ylabel("Count")
    ax.set_title("Stage 3A pilot — region 03 (dev only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "error_histogram.png", dpi=150)
    plt.close(fig)

    # success / failure panels
    sorted_rows = sorted(pilot_rows, key=lambda r: r["geodesic_error_m"])
    examples = sorted_rows[:2] + sorted_rows[-2:]
    for i, r in enumerate(examples):
        tag = "success" if r["geodesic_error_m"] <= 100 else "failure"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        qpath = Path(r["_path"])
        qim = cv2.cvtColor(cv2.imread(str(qpath)), cv2.COLOR_BGR2RGB)
        axes[0].imshow(qim)
        axes[0].set_title(f"Query {r['filename']}\nerr={r['geodesic_error_m']:.1f}m ({tag})")
        axes[0].axis("off")
        tid = int(r["predicted_tile_id"])
        if tid >= 0:
            t = tiles[tid]
            patch = np.asarray(sat[t.row0 : t.row1, t.col0 : t.col1, :3])
            axes[1].imshow(patch)
            axes[1].set_title(f"Pred tile {t.tile_id} centre")
        else:
            sh, sw = sat.shape[:2]
            pr, pc = lonlat_to_pixel(r["pred_lon"], r["pred_lat"], georef)
            r0, c0 = max(0, int(pr) - 256), max(0, int(pc) - 256)
            patch = np.asarray(sat[r0 : r0 + 512, c0 : c0 + 512, :3])
            axes[1].imshow(patch)
            axes[1].set_title("Pred location patch (map)")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(FIGS / f"example_{i+1}_{tag}.png", dpi=120)
        plt.close(fig)


def main() -> int:
    cfg = load_config()
    region = cfg["leakage_and_splits"]["pilot_region"]
    thresholds = [t["value"] for t in cfg["success_thresholds_m"]]

    with (ROOT / "datasets" / "manifests" / "satellite_metadata.csv").open(encoding="utf-8", newline="") as f:
        meta_rows = list(csv.DictReader(f))
    sat_meta = sat_row_for_region(meta_rows, region)
    sat_path = Path(sat_meta["satellite_path"])
    west, north, sx, sy = georef_from_metadata(sat_meta)
    georef = (west, north, sx, sy)

    spec = TileSpec(
        int(cfg["satellite_tiling"]["tile_size_px"]),
        int(cfg["satellite_tiling"]["stride_px"]),
    )
    sat, (h, w) = load_sat_memmap(sat_path)
    tiles = build_tiles(h, w, west, north, sx, sy, spec)
    cpx = int(cfg["matcher_params"]["coarse_ncc"]["coarse_patch_px"])
    print(f"Precomputing coarse tile fingerprints ({len(tiles)} tiles)...", flush=True)
    tile_coarse = precompute_tile_coarse(sat, tiles, cpx)
    print(f"Region {region}: {h}x{w} px, {len(tiles)} tiles", flush=True)

    manifest = load_manifest(region)
    dev_n = int(cfg["leakage_and_splits"]["matcher_dev_subset"]["n_queries"])
    dev_seed = int(cfg["leakage_and_splits"]["matcher_dev_subset"]["seed"])
    pilot_n = int(cfg["pilot"]["n_queries"])
    pilot_seed = 20260923

    dev_queries = select_queries(manifest, dev_n, dev_seed)
    pilot_queries = select_queries(manifest, pilot_n, pilot_seed)

    # Dev-only matcher comparison
    dev_results = {}
    for mid in [c["id"] for c in cfg["matcher_candidates_dev_only"]]:
        print(f"Dev eval: {mid} ...", flush=True)
        dev_results[mid] = evaluate_matcher(
            mid, dev_queries, sat, tiles, georef, cfg, thresholds, tile_coarse
        )

    # Select: maximize success@100m then minimize median error
    def rank_key(mid: str):
        sr = dev_results[mid]["success_rates"].get("100", 0)
        med = dev_results[mid]["median_error_m"] or 1e9
        return (sr, -med)

    selected = max(dev_results.keys(), key=rank_key)
    cfg["matcher_selected"] = selected
    cfg["frozen_at"] = datetime.now(timezone.utc).isoformat()
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # Pilot with frozen matcher
    pilot_rows = []
    for q in pilot_queries:
        path = ROOT / "datasets" / "raw" / "UAV_VisLoc_dataset" / region / "drone" / q["filename"]
        t0 = time.perf_counter()
        bgr = cv2.imread(str(path))
        max_long = int(cfg["query_preprocessing"]["max_long_edge_px"])
        if selected == "pyramid_ncc_rot4":
            p = cfg["matcher_params"]["pyramid_ncc_rot4"]
            sat_rgb = np.asarray(sat[:, :, :3])
            pred_lat, pred_lon, score, margin, fail = pyramid_ncc_predict(
                bgr,
                sat_rgb,
                georef,
                int(p["max_long_edge_px"]),
                int(p["satellite_max_long_edge_px"]),
                tuple(int(x) for x in p["rotations_deg"]),
            )
            tile_id = -1
        elif selected == "akaze_coarse_to_fine":
            pred_lat, pred_lon, tile_id, score, margin, fail = _akaze_predict_lonlat(
                bgr,
                sat,
                tiles,
                georef,
                cfg["matcher_params"]["akaze_coarse_to_fine"],
                max_long,
                tile_coarse,
            )
        else:
            _, qg = _prep_query(bgr, max_long)
            scores = coarse_ncc_scores(
                qg,
                sat,
                tiles,
                cfg["matcher_params"]["coarse_ncc"]["coarse_patch_px"],
                tile_coarse,
            )
            order = np.argsort(-scores)
            t = tiles[int(order[0])]
            tile_id = t.tile_id
            pred_lat, pred_lon = t.center_lat, t.center_lon
            score = float(scores[order[0]])
            margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0
            fail = None
        runtime = time.perf_counter() - t0
        gt_lat, gt_lon = float(q["lat"]), float(q["lon"])
        err = geodesic_m(gt_lon, gt_lat, pred_lon, pred_lat)
        row = {
            "query_id": q["num"],
            "region": region,
            "filename": q["filename"],
            "gt_lat": gt_lat,
            "gt_lon": gt_lon,
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "predicted_tile_id": tile_id,
            "top_score": round(score, 4),
            "score_margin": round(margin, 4),
            "geodesic_error_m": round(err, 3),
            "success_50m": int(err <= 50),
            "success_100m": int(err <= 100),
            "success_250m": int(err <= 250),
            "success_500m": int(err <= 500),
            "runtime_s": round(runtime, 3),
            "failure_reason": fail or "",
            "matcher_id": selected,
            "search_used_gt_position": False,
            "_path": str(path),
        }
        pilot_rows.append(row)

    csv_path = REPORTS / "stage3a_pilot_results.csv"
    write_pilot_csv(pilot_rows, csv_path)

    errs = [r["geodesic_error_m"] for r in pilot_rows]
    summary = {
        "generated_at": cfg["frozen_at"],
        "matcher_frozen": selected,
        "dev_eval": dev_results,
        "pilot_n": len(pilot_rows),
        "pilot_median_error_m": float(np.median(errs)),
        "pilot_mean_error_m": float(np.mean(errs)),
        "success_rates": {
            str(t): sum(r[f"success_{t}m"] for r in pilot_rows) / len(pilot_rows) for t in thresholds
        },
        "n_failures_ge_100m": sum(1 for e in errs if e > 100),
        "n_success_le_100m": sum(1 for e in errs if e <= 100),
        "leakage_checks": {
            "search_used_gt_position": False,
            "n_tiles_searched": len(tiles),
            "full_map_search_per_query": True,
        },
    }
    (REPORTS / "stage3a_pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_figures(pilot_rows, tiles, sat, cfg, georef)
    write_markdown_reports(cfg, dev_results, selected, pilot_rows, summary, len(tiles))
    print(json.dumps(summary, indent=2))
    return 0


def write_markdown_reports(cfg, dev_results, selected, pilot_rows, summary, n_tiles):
    """Update machine-readable gate JSON only.

    Hand-reviewed narratives live in reports/stage3a_matcher_protocol.md and
    reports/stage3a_pilot_summary.md — do not overwrite them from this runner.
    """
    del cfg, dev_results, selected, pilot_rows, n_tiles  # reserved for future auto stubs
    sr100 = summary["success_rates"].get("100", 0)
    sr500 = summary["success_rates"].get("500", 0)
    summary["leakage_checks"]["both_success_and_failure_100m"] = 0 < sr100 < 1
    summary["leakage_checks"]["both_success_and_failure_500m"] = 0 < sr500 < 1
    summary["leakage_checks"]["both_success_and_failure"] = summary["leakage_checks"][
        "both_success_and_failure_100m"
    ]
    gate = (
        "READY"
        if summary["leakage_checks"]["both_success_and_failure_100m"]
        else "NOT READY"
    )
    summary["full_label_generation_gate"] = gate
    (REPORTS / "stage3a_pilot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Stage 3A full-label gate: {gate}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
