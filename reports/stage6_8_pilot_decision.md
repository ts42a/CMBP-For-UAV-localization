# Stages 6-8 pilot — decision record (function check)

**Scope:** Stage 5 pilot labels, 900 attempts (90 frames, 10 per region; region 07 excluded by D2),
protocol `configs/predictor_protocol_v1.yaml` frozen at commit `28fb7e6` before any predictor ran.
This is a function check of the pipeline, **not** a geographic result: each held-out group has
10-20 frames, and some have only one or two frames in the minority class.
Full tables: `reports/stage6_8_pilot_summary.md`. Outputs: `results/stage6_7/pilot/`, `results/stage8/pilot/`.

## Pipeline checks

| Check | Result |
|---|---|
| Features use exactly the localizer's windows | pass: `n_window_tiles` reproduced on all 900 rows (asserted) |
| Predictor inputs map-only (prior centre, radius, map) | pass: no ground truth, UAV image or localizer output in `features.csv` |
| No test-group row in any scaler / PCA / model / calibrator fit | pass (asserted in every fold) |
| All 7 folds from one command, one protocol hash | pass (~1 min features cached; ~3.5 min features from scratch) |
| Distinctiveness does not simply track window size | pass: competitors grow ~10x (median 21 → 196) from R=100 to 1000 m, `distinct_max` median 0.135 → 0.112 (`figures/stage7_pilot/distinct_vs_radius.png`) |
| Shuffle control near 0.5 | noisy: macro AUROC 0.42 (per fold 0.12-0.58); the extreme folds have a single minority frame (G4: 5 positive rows, G7: 10 negative rows), so a one-seed control is not informative at pilot size |

## What the numbers show (raw predictions, within held-out groups)

| model | macro AUROC | macro AUROC excl. G2 (dev) |
|---|---:|---:|
| M1 texture | 0.55 | 0.54 |
| M2 DINO | 0.65 | 0.66 |
| M3 DINO + distinctiveness | 0.64 | 0.64 |

DINO map features rank attempts within a held-out region somewhat better than texture; adding
distinctiveness does not help at this size. At 10 frames per region none of this is a finding.

## Problem found: source-fitted calibration inverts under leave-one-group-out

Calibrated AUROC equals 1 - raw AUROC in several folds (G2-G4), which a monotone calibrator can
only produce with a negative slope. Diagnosis (read-only, on source out-of-fold predictions):

- Within each source group, OOF predictions rank above chance (mean within-group AUROC
  0.49-0.65 texture, 0.62-0.69 DINO).
- Across source groups, the group-mean OOF prediction is **anti-correlated** with the group's
  success rate (constant model r = -0.93 to -0.97; texture down to -0.73). Leaving group g out
  makes the training prevalence move opposite to g's own prevalence, and region success rates
  range from 0.00 (region 10) to 0.94 (region 03), far more than the map features explain.
- Platt is fitted on the pooled OOF predictions, so this between-group term dominates. Its slope
  comes out near zero or negative (-0.33 to +0.49), calibrated predictions collapse to a narrow
  band, and calibrated Brier / log loss are worse than raw on most folds.
- The same artefact makes **pooled** AUROC meaningless here (constant model 0.155). Macro
  (within-group) AUROC is the ranking metric to read.

This is a property of the frozen protocol under large between-region prevalence differences,
not an implementation bug. Under the plan's rule, it is not fixed on these results by editing
`predictor_protocol_v1.yaml`. Any change must be a dated `predictor_protocol_v2` written
**before the full-scale labels exist**, logged as a deviation, with the reason cited to this record.

## Decisions needed before the full run (not acted on)

1. **Calibration design.** Options to consider for v2: constrain the Platt slope to be
   non-negative; fit the calibrator within source groups (group-centred scores) rather than
   pooled; or report calibration as not transferable (plan Stage 8 fallback: "report ranking
   transfer without it"). The between-region prevalence spread itself may be the headline
   finding for RQ2.
2. **Pooled metrics.** Treat pooled AUROC and PR-AUC as unreliable under this split and report
   macro and per-group instead. Pooled Brier and log loss remain interpretable.
3. **Shuffle control.** At full scale, one seed is probably enough; if not, average several
   permutations, declared before the run.

## Status
Stages 6-8 pipeline runs end to end and passes its leakage checks. Calibration needs a protocol
revision before the full run. Next in plan order: the Stage 10 Kaggle benchmark (the full
labelling run needs ~150-200 CPU-hours) and H1 synthetic queries.
