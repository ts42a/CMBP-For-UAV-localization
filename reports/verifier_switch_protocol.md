# Verifier switch protocol — AKAZE vs DINOv2 patch matching (declared before any run)

Declared 2026-09-27, before the patch verifier was run on any frame. Committed to git
before results exist. Nothing below is changed after results are seen.

## Why
Stage 4 zero-offset test failed (0/5) and the region-03 crop-width diagnosis showed
AKAZE+RANSAC cross-view verification fails at every crop width
(`reports/stage4_checks.md`, `reports/stage4_verify_scale_diagnosis.md`).
The plan allows one localizer change, documented, selected without test regions.

## What changes, what stays frozen
- Unchanged: DINOv2 ViT-S/14 retrieval, 1024/512 tiling, 4-rotation retrieval bank,
  top-k 20, windowing, offsets, tolerances, splits.
- Changed: the fine verification step only. AKAZE+homography is replaced by
  DINOv2 ViT-S/14 patch tokens (same frozen weights) + mutual nearest neighbours +
  RANSAC similarity transform. Fixed design (not tuned):
  query long edge 448 px; query rotations every 45 deg (8); satellite context crop
  centred on each candidate tile, resampled to 448x448; RANSAC reprojection threshold
  14 px (one patch); plausibility: similarity scale within [0.25, 4] and mapped
  query centre inside the crop; best candidate = most inliers.
- Tuned on region-03 tune frames only (24 frames): context crop width as a multiple of
  the tile, s in {1, 2, 3} or multi-scale {2, 3}; and the min-inlier accept threshold in
  {10, 15, 20, 30}. Selection uses the pre-registered criterion: highest precision among
  settings with >= 5 accepted-correct; ties -> more accepted-correct.

## Comparison data (region 03 tune frames only)
- A. Windowed attempts: the same 216 pre-registered attempts (24 frames x 3 radii x 3
  offsets) already run with AKAZE (`results/stage4/tune`).
- B. Zero-offset, R = 100 m, all 24 tune frames, both verifiers.
- Visual check: panels of every accepted fix from both verifiers (query | map crop at
  the prediction | map crop at the truth). Scene judged correct only when distinctive
  structures (roads, rivers, buildings, field boundaries) visibly correspond. Reviewer:
  Claude, from the rendered panels; recorded in a CSV with a note per panel.

## "Improves reliably" — all must hold, else stop (no re-freeze)
1. Accepted-correct on A >= 2x AKAZE's (AKAZE: 7 -> need >= 14).
2. Precision on A >= 0.90.
3. Accepted-correct on A spread over >= 8 distinct frames of 24.
4. Zero-offset B: accepted-correct >= 12/24.
5. Visual: >= 90% of accepted-correct panels show corresponding scene content.
Positional error of accepted fixes (median, p90) is reported for both verifiers.

## If it improves: re-freeze, then gates on untouched data
New frozen config `configs/matcher_stage4_frozen.yaml`, deviation logged in
`configs/preregistration_v1.yaml`. Then:

**Stage 3B validation gate rerun** — region-03 val (30 frames, never used for tuning),
full-map search as originally. PASS requires all of:
- accepted-correct >= 5/30 at 100 m;
- precision among accepted >= 0.80;
- at least one rejected or accepted-incorrect frame (a mix);
- search never used CSV position; no prediction is an exact CSV copy;
- visual: >= 80% of accepted-correct panels show corresponding scene content.

**Stage 4 checks rerun** — `scripts/protocol/stage4_checks.py` unchanged criteria:
zero-offset >= 3/5, determinism, 2 km negative control, kill/resume.

Only if both pass: Stage 5 multi-region pilot (`label_windowed.py --set pilot`).
If either fails: stop and report. No switch to SuperPoint+LightGlue without a separate decision.
