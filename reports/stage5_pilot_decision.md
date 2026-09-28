# Stage 5 pilot — decision record

**Gate (pre-registered `stage5_go`, per `reports/stage5_go_protocol.md`): GO**

Run: `results/stage5/pilot/outcomes.csv` — 110 frames (10 per region x 11), 3 radii x 3 offsets,
990 attempts; commit `7c1edc9`, clean tree, one config hash (`matcher_stage4_frozen.yaml`),
accept rule `patch_20`, tolerance 100 m. Full tables: `reports/stage5_pilot_summary.md`.

| Gating check | Value | Pass |
|---|---|---|
| Pooled accepted-correct rate in [0.15, 0.85] | 0.447 (442/990) | yes |
| Accepted-correct non-increasing across offset bins | 0.52 → 0.44 → 0.28 → 0.17 | yes |

Informational: accepted-incorrect 9/990 (precision 0.98); pooled excluding dev region 03: 0.397;
secondary tolerances 25 m 0.20, 50 m 0.39; radius trend not monotonic (0.45 / 0.49 / 0.40 at
100 / 300 / 1000 m), as anticipated in `stage5_go_protocol.md` Section 4.

## By-eye inspection (plan Stage 5 check)

**Every accepted-incorrect row** (9 rows, 2 frames, all at exactly 20 inliers) —
`figures/stage5_pilot_incorrect/`, review in `reports/stage5_pilot_visual_review.csv`:
- `05_0236` (Yunnan, 3 rows, 804 m): forested slope with eroded ground, no distinctive
  structure; truth near the map edge. Genuine wrong fix on self-similar terrain.
- `06_0172` (Zhuxi, 6 rows, 347–363 m): the map **at the prediction** shows the query scene
  (curving highway W, long row of grey-roof blocks, small blue roof, algae-green lake E); the map
  at the CSV position (dense village, no lake) does not. Likely a **ground-truth offset**, not a
  matcher error.

**Sample of held-out-region accepted-correct** (one per region with positives, 8 panels,
`figures/stage5_pilot_correct_sample/`): 8/8 show corresponding scenes.

## Findings that affect Stages 6-9 (not acted on — decisions needed)

1. **Group G5 (regions 07 + 10) has zero positives**: 0/180 accepted. With G5 as the test
   fold, AUROC, PR-AUC and calibration are undefined. By-eye check (region 07 flagged in
   progress.md limitation 6):
   - Region 07: the map is a ~50 m x 900 m strip (170 x 3000 px); each low-altitude drone frame
     covers a full village block, so the map does not contain the query footprint. Unusable for
     map-based localization on map geometry alone (already flagged in Stage 2), independent of
     outcomes.
   - Region 10: low-altitude close-ups (a wind turbine fills much of the frame) vs 600-900 m
     verifier context crops tuned on region 03's ~466 m flights, plus seasonal change. Fits land
     near truth (median raw error 90 m) but top out at 18 inliers. Scale mismatch from skipping
     height normalisation (height datum unresolved).
2. **Accept threshold transfers unevenly**: region 03 (tune) fits reach 30-70 inliers; regions
   06, 07, 10 top out at 15-20. Per-region accepted-correct ranges 0.00-0.94. Changing the
   threshold using these regions would be tuning on test regions.
3. **Possible along-track ground-truth offsets**: for region-06 frames whose fits (>= 16
   inliers) land > 100 m from the CSV position, 3 of 4 offsets point along the flight track
   (e.g. 06_0172: offset bearing 164 deg, track 164 deg; 06_0103: 330 vs 344). Consistent with
   a GPS/camera timing lag or forward camera tilt. Region 05 offsets do not align with the
   track. Small sample (4-5 frames): a hypothesis to audit, not a finding.
4. **Dev-region optimism**: region 03 scores 0.94 vs 0.40 for the other regions pooled.

## Status
Stage 5 pilot GO. Plan order next: Stages 6-8 pilot on these labels (function check), then
the Stage 10 Kaggle benchmark. Findings 1-3 need a decision first (see progress.md).
