# Stage 3B — Decision

**Decision: READY** for full real-outcome label generation with frozen matcher **`dinov2_c2f`**.

## Exact evidence

### 1. Untouched validation subset (frozen before tuning)

- Split file: `configs/stage3b_splits.yaml` (`frozen_at=2026-09-24T03:16:52.758184+00:00`)
- `n_val=30`, `n_tune=24`, tune∩val = empty
- Hyperparameters chosen on **tune only**; gate uses **val only**

| Candidate | Val n | @100 m succ/fail | @500 m succ | Median err (m) | Top-k recall | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| dinov2_global | 30 | 7/23 | 8 | 2161 | 0.57 | 3.24 |
| **dinov2_c2f (selected)** | 30 | **9/21** | 13 | 1178 | 0.57 | 10.35 |

### 2. Visual credibility (human review of val panels)

Reviewed figures under `figures/stage3b/val_dinov2_c2f_*`. Table: `reports/stage3b_val_human_visual_review.csv`.

- **Credible @100 m successes inspected:** 5 (including `03_0257` 18.8 m, `03_0709` 23.4 m, `03_0168` 11.7 m, `03_0441` 40.4 m, `03_0326` 50.7 m) — pred patches match GT scene content.
- **Clear failures inspected:** 1+ (e.g. `03_0014` 5654 m — urban bridge query → empty grassland pred).
- Positional-only chance hits without scene agreement were the Stage 3A failure mode; these val successes are **not** of that type.

Freeze rule met: useful mix of **visually credible successes and failures at 100 m** on untouched val.

### 3. Stage 3A baseline preserved (comparison set, not used for selection)

| Mode | n | @100 m | @500 m | Median err (m) | Top-k recall | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| pyramid_ncc_rot4 (baseline) | 100 | 0 | 2 | 3296 | n/a | 2.24 |
| dinov2_global | 100 | 8 | 16 | 2096 | 0.44 | 0.98 |
| dinov2_c2f | 100 | 13 | 31 | 1579 | 0.44 | 3.83 |

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
py -3 F:\CMBL\scripts\localization\freeze_stage3b_splits.py
py -3 F:\CMBL\scripts\run_stage3b_recovery.py
py -3 F:\CMBL\scripts\localization\finalize_stage3b_decision.py
```
