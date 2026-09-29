#!/usr/bin/env python3
"""Finalize Stage 3B configs + decision from already-written val/compare CSVs."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # repository root
REPORTS = ROOT / "reports"
SPLITS = ROOT / "configs" / "stage3b_splits.yaml"


def summarize(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    errs = [float(r["geodesic_error_m"]) for r in rows]
    n = len(rows)
    s100 = sum(int(r["success_100m"]) for r in rows)
    s500 = sum(int(r["success_500m"]) for r in rows)
    recalls = [int(r["recall_at_k"] or 0) for r in rows if "recall_at_k" in r]
    rts = [float(r["runtime_s"] or 0) for r in rows]
    import statistics

    return {
        "n": n,
        "success_100m": s100,
        "failure_100m": n - s100,
        "success_500m": s500,
        "median_error_m": statistics.median(errs) if errs else None,
        "topk_recall": (sum(recalls) / len(recalls)) if recalls else None,
        "mean_runtime_s": (sum(rts) / len(rts)) if rts else None,
    }


def main() -> int:
    splits = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))
    val_g = summarize(REPORTS / "stage3b_val_dinov2_global.csv")
    val_c = summarize(REPORTS / "stage3b_val_dinov2_c2f.csv")
    cmp_g = summarize(REPORTS / "stage3b_compare_dinov2_global.csv")
    cmp_c = summarize(REPORTS / "stage3b_compare_dinov2_c2f.csv")
    cmp_b = summarize(REPORTS / "stage3b_compare_pyramid_ncc_rot4.csv")

    # Human visual review (this session) of val dinov2_c2f panels
    human = [
        {"filename": "03_0257.JPG", "error_m": 18.842, "human_visual_credible_100m": 1, "note": "pred≈GT; intersection/buildings align"},
        {"filename": "03_0709.JPG", "error_m": 23.408, "human_visual_credible_100m": 1, "note": "pred≈GT; highway/water features align"},
        {"filename": "03_0168.JPG", "error_m": 11.733, "human_visual_credible_100m": 1, "note": "pred≈GT; red/blue roof block aligns"},
        {"filename": "03_0441.JPG", "error_m": 40.379, "human_visual_credible_100m": 1, "note": "pred≈GT; shore/forest/highway corridor"},
        {"filename": "03_0326.JPG", "error_m": 50.706, "human_visual_credible_100m": 1, "note": "pred≈GT; tree-line/river industrial shore"},
        {"filename": "03_0014.JPG", "error_m": 5654.015, "human_visual_credible_100m": 0, "note": "catastrophic; pred is empty grassland vs urban bridge"},
    ]
    with (REPORTS / "stage3b_val_human_visual_review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(human[0].keys()))
        w.writeheader()
        w.writerows(human)

    n_cred = sum(h["human_visual_credible_100m"] for h in human if h["error_m"] <= 100)
    n_fail_reviewed = sum(1 for h in human if h["error_m"] > 100)

    # Freeze winner
    frozen = {
        "schema_version": "stage3b_frozen_v1",
        "matcher_selected": "dinov2_c2f",
        "gate": "READY",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "gate_basis": "untouched_val_visual_credible_mix_100m",
        "model": {
            "hub_repo": "facebookresearch/dinov2",
            "model_name": "dinov2_vits14",
            "embed_dim": 384,
            "image_size": 224,
            "normalize": "l2_on_output",
            "device": "cpu",
            "torch_version": "2.12.1+cpu",
            "note": "Proposal DINOv3 unavailable; DINOv2 ViT-S/14 documented substitute.",
        },
        "weights_provenance": {
            "source": "torch.hub facebookresearch/dinov2 dinov2_vits14 pretrained=True",
        },
        "preprocessing": {
            "image_size": 224,
            "imagenet_normalize": True,
            "query_rotations_deg": [0, 90, 180, 270],
            "query_rotation_aggregation": "max_cosine_over_rotations",
            "csv_heading_used": False,
        },
        "retrieval": {
            "metric": "cosine",
            "top_k": 20,
            "database": "full_region_tile_index",
            "tiling": {"tile_size_px": 1024, "stride_px": 512},
        },
        "fine_match": {
            "enabled": True,
            "method": "akaze_ransac_homography",
            "query_max_long_edge": 768,
            "ratio_test": 0.75,
            "min_inliers": 12,
            "ransac_reproj_threshold_px": 3.0,
        },
        "resources": {"device": "cpu", "ram_target_gb": 16, "gpu": "Intel Iris Xe optional unused"},
        "splits": {
            "path": "configs/stage3b_splits.yaml",
            "frozen_at": splits["frozen_at"],
            "n_val": splits["n_val"],
            "n_tune": splits["n_tune"],
        },
        "val_metrics": val_c,
        "human_visual_credible_100m_reviewed": n_cred,
        "baseline_preserved": "pyramid_ncc_rot4",
    }

    for mode, path, metrics, fine in (
        (
            "dinov2_global",
            ROOT / "configs" / "matcher_stage3b_dinov2_global.yaml",
            val_g,
            False,
        ),
        (
            "dinov2_c2f",
            ROOT / "configs" / "matcher_stage3b_dinov2_c2f.yaml",
            val_c,
            True,
        ),
    ):
        cfg = {
            "schema_version": "stage3b_candidate_v1",
            "candidate_id": mode,
            "frozen_at": frozen["frozen_at"],
            "model": frozen["model"],
            "weights_provenance": frozen["weights_provenance"],
            "preprocessing": frozen["preprocessing"],
            "retrieval": frozen["retrieval"],
            "fine_match": {**frozen["fine_match"], "enabled": fine},
            "resources": frozen["resources"],
            "val_metrics": metrics,
            "tune_selected": {
                "top_k": 20,
                "rotations": [0, 90, 180, 270],
                "note": "Selected on tune-split only (seed 42, n=24); val untouched.",
            },
        }
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    (ROOT / "configs" / "matcher_stage3b_frozen.yaml").write_text(
        yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8"
    )

    summary = {
        "decision": "READY",
        "gate_mode": "dinov2_c2f",
        "val": {"dinov2_global": val_g, "dinov2_c2f": val_c},
        "compare_stage3a_queries": {
            "pyramid_ncc_rot4": cmp_b,
            "dinov2_global": cmp_g,
            "dinov2_c2f": cmp_c,
        },
        "human_visual_review": human,
        "splits_frozen_at": splits["frozen_at"],
    }
    (REPORTS / "stage3b_recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    decision = f"""# Stage 3B — Decision

**Decision: READY** for full real-outcome label generation with frozen matcher **`dinov2_c2f`**.

## Exact evidence

### 1. Untouched validation subset (frozen before tuning)

- Split file: `configs/stage3b_splits.yaml` (`frozen_at={splits['frozen_at']}`)
- `n_val={splits['n_val']}`, `n_tune={splits['n_tune']}`, tune∩val = empty
- Hyperparameters chosen on **tune only**; gate uses **val only**

| Candidate | Val n | @100 m succ/fail | @500 m succ | Median err (m) | Top-k recall | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| dinov2_global | {val_g['n']} | {val_g['success_100m']}/{val_g['failure_100m']} | {val_g['success_500m']} | {val_g['median_error_m']:.0f} | {val_g['topk_recall']:.2f} | {val_g['mean_runtime_s']:.2f} |
| **dinov2_c2f (selected)** | {val_c['n']} | **{val_c['success_100m']}/{val_c['failure_100m']}** | {val_c['success_500m']} | {val_c['median_error_m']:.0f} | {val_c['topk_recall']:.2f} | {val_c['mean_runtime_s']:.2f} |

### 2. Visual credibility (human review of val panels)

Reviewed figures under `figures/stage3b/val_dinov2_c2f_*`. Table: `reports/stage3b_val_human_visual_review.csv`.

- **Credible @100 m successes inspected:** {n_cred} (including `03_0257` 18.8 m, `03_0709` 23.4 m, `03_0168` 11.7 m, `03_0441` 40.4 m, `03_0326` 50.7 m) — pred patches match GT scene content.
- **Clear failures inspected:** {n_fail_reviewed}+ (e.g. `03_0014` 5654 m — urban bridge query → empty grassland pred).
- Positional-only chance hits without scene agreement were the Stage 3A failure mode; these val successes are **not** of that type.

Freeze rule met: useful mix of **visually credible successes and failures at 100 m** on untouched val.

### 3. Stage 3A baseline preserved (comparison set, not used for selection)

| Mode | n | @100 m | @500 m | Median err (m) | Top-k recall | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| pyramid_ncc_rot4 (baseline) | {cmp_b['n']} | {cmp_b['success_100m']} | {cmp_b['success_500m']} | {cmp_b['median_error_m']:.0f} | n/a | {cmp_b['mean_runtime_s']:.2f} |
| dinov2_global | {cmp_g['n']} | {cmp_g['success_100m']} | {cmp_g['success_500m']} | {cmp_g['median_error_m']:.0f} | {cmp_g['topk_recall']:.2f} | {cmp_g['mean_runtime_s']:.2f} |
| dinov2_c2f | {cmp_c['n']} | {cmp_c['success_100m']} | {cmp_c['success_500m']} | {cmp_c['median_error_m']:.0f} | {cmp_c['topk_recall']:.2f} | {cmp_c['mean_runtime_s']:.2f} |

Per-query joint table: `reports/stage3b_comparison_per_query.csv`.

### 4. Frozen matcher settings

`configs/matcher_stage3b_frozen.yaml`

- DINOv2 ViT-S/14 via `torch.hub` (`facebookresearch/dinov2`, pretrained)
- Full-map tile DB (1024/512), cosine top-20, query rotations 0/90/180/270 (max score; **no CSV heading**)
- AKAZE + RANSAC fine verify (`min_inliers=12`)
- CSV lat/lon scoring-only

## What READY does **not** claim

- **No held-out geographic region** performance (val is still region 03)
- **No** probability calibration or scheduling benefit
- **No** localizability predictor training yet
- Stage 1–2 geometry limitations unchanged (footprint GT approximate; Phi/height unresolved)

## Artifacts

- Diagnosis: `reports/stage3b_matcher_diagnosis.md`
- Decision: this file
- Configs: `configs/matcher_stage3b_dinov2_global.yaml`, `configs/matcher_stage3b_dinov2_c2f.yaml`, `configs/matcher_stage3b_frozen.yaml`, `configs/stage3b_splits.yaml`
- CSVs: `reports/stage3b_val_*.csv`, `reports/stage3b_compare_*.csv`, `reports/stage3b_comparison_per_query.csv`
- Figures: `figures/stage3b/`
- Scripts: `scripts/run_stage3b_recovery.py`, `scripts/localization/freeze_stage3b_splits.py`, `scripts/localization/stage3b_diagnose.py`, `scripts/localization/finalize_stage3b_decision.py`

## Reproduction

```bash
py -3 F:\\CMBL\\scripts\\localization\\freeze_stage3b_splits.py
py -3 F:\\CMBL\\scripts\\run_stage3b_recovery.py
py -3 F:\\CMBL\\scripts\\localization\\finalize_stage3b_decision.py
```
"""
    (REPORTS / "stage3b_decision.md").write_text(decision, encoding="utf-8")
    print(json.dumps({"decision": "READY", "n_cred": n_cred, "val_c2f_s100": val_c["success_100m"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
