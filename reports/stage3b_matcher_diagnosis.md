# Stage 3B — Matcher diagnosis (from Stage 3A baseline)

**Baseline frozen matcher:** `pyramid_ncc_rot4` (`configs/matcher_protocol.yaml`)  
**Baseline evidence:** 0/100 success@100 m; median error ~3.3 km; 2 positional @500 m hits visually wrong  
(`reports/stage3a_pilot_summary.md`, `figures/stage3a/`).

Stage 1–2 limitations preserved: no validated footprint GT; Phi/height/intrinsics unresolved;
CSV lat/lon usable for centres / scoring only; do not use unknown footprint geometry as GT.

## What failed (representative sample, n=8)

Sample selection: 2 lowest / 2 mid / 4 highest Stage 3A geodesic errors on region 03.
CSV used only for oracle diagnostics and scoring — never for candidate selection.

| Check | Finding |
|---|---|
| Satellite coverage | **8/8** sample GT centres on `satellite03.tif` |
| Tile grid | GT centres fall inside ≥1 overlapping 1024 px tile when on-map |
| Scale / intrinsics | Unresolved (Stage 2D); Stage 3A used fixed resize only — plausible contributor |
| Orientation | Stage 3A used 0/90/180/270 bank; vehicle vs camera yaw still unresolved |
| Appearance / season | UAV oblique vs ortho satellite; colour/texture shift; NCC at GT often weak |
| Ranking | NCC peak frequently far from GT; aliasing to similar rural texture |
| Correct area in top candidates | DINOv2 top-20 recall on sample: **6/8** (embedding path only; diagnostic) |

### Per-sample table

See `reports/stage3b_diagnosis_sample.csv`. Panels: `figures/stage3b/diag_*.png`.

### Codes observed

- `catastrophic_alias`: 6
- `retrieval_ok_localization_fail`: 4
- `weak_crossmodal_ncc_at_gt`: 3
- `ncc_prefers_wrong_or_tied`: 3
- `embedding_miss_top20`: 2

## Interpretation

1. **Not a coverage bug.** GT centres are on-map; nearest tile centres are within ~half-tile of GT.
2. **Cross-modal NCC is a weak score** at the true location (`weak_crossmodal_ncc_at_gt` / `ncc_prefers_wrong_or_tied`).
   Global template matching therefore aliases to distant similar fields/roads.
3. **Orientation / scale** remain under-constrained; fixed cardinal rotations and fixed resize are insufficient
   for oblique UAV → ortho map matching.
4. **Embedding retrieval is a plausible recovery path:** when DINOv2 cache is present, GT neighbourhood
   sometimes appears in top-20 even when Stage 3A NCC failed catastrophically — motivating pretrained
   global descriptors + coarse-to-fine geometric verification (Stage 3B candidates).
5. Stage 3A @500 m “successes” remain **visually non-credible** (Stage 3A figures) — do not widen
   the primary 100 m threshold to manufacture labels.

## Implications for Stage 3B candidates

| Candidate | Rationale |
|---|---|
| **A. DINOv2 ViT-S/14 global retrieval** | Stronger cross-view descriptors than NCC; fits 16 GB / CPU |
| **B. DINOv2 + AKAZE/RANSAC fine verify** | Coarse-to-fine with geometric verification; no camera PnP / footprint GT |

Proposal compact DINOv3 remains unavailable; DINOv2 ViT-S/14 is the documented substitute.

## Reproduction

```bash
py -3 F:\CMBL\scripts\localization\stage3b_diagnose.py
```
