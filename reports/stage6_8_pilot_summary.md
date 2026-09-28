# stage6_8_pilot — Stages 6-8 predictor summary

**Function check on the Stage 5 pilot labels (10 frames per region), not a geographic result.** Protocol `configs/predictor_protocol_v1.yaml` (sha256 `90efed076da2…`), fixed hyperparameters, region 07 excluded (deviation D2). Rows 900, prevalence 0.491.

## Pooled over held-out groups (95% CI: region-group bootstrap) and macro over folds

| model | variant | pooled AUROC (CI) | pooled Brier (CI) | pooled log loss | pooled ECE | macro AUROC | macro AUROC excl. G2 | macro Brier |
|---|---|---|---|---:|---:|---:|---:|---:|
| M0_constant | raw | 0.155 (0.074–0.342) | 0.285 (0.261–0.314) | 0.763 | 0.323 | 0.500 | 0.500 | 0.285 |
| M1_texture | raw | 0.427 (0.278–0.631) | 0.317 (0.267–0.374) | 0.868 | 0.264 | 0.552 | 0.536 | 0.320 |
| M1_texture | calibrated | 0.199 (0.090–0.428) | 0.298 (0.272–0.325) | 0.799 | 0.307 | 0.445 | 0.461 | 0.301 |
| M2_dino | raw | 0.597 (0.393–0.793) | 0.291 (0.192–0.415) | 0.940 | 0.174 | 0.651 | 0.658 | 0.287 |
| M2_dino | calibrated | 0.228 (0.071–0.543) | 0.318 (0.265–0.389) | 0.850 | 0.313 | 0.421 | 0.428 | 0.328 |
| M3_dino_distinct | raw | 0.587 (0.359–0.791) | 0.295 (0.191–0.424) | 0.955 | 0.192 | 0.639 | 0.643 | 0.293 |
| M3_dino_distinct | calibrated | 0.212 (0.070–0.495) | 0.320 (0.265–0.396) | 0.852 | 0.317 | 0.423 | 0.431 | 0.329 |

Shuffle control (M1, labels permuted within training groups): macro AUROC 0.419 (per fold: G1 0.545, G2 0.583, G3 0.496, G4 0.118, G5 —, G6 0.480, G7 0.294).

## Per held-out group

| fold | regions | model | variant | n | prev. | AUROC | PR-AUC | Brier | log loss | ECE | calibrator |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| G1 | 01,02 | M0_constant | raw | 180 | 0.38 | 0.500 | 0.378 | 0.255 | 0.703 | 0.142 |  |
| G1 | 01,02 | M1_texture | raw | 180 | 0.38 | 0.689 | 0.494 | 0.327 | 0.895 | 0.343 |  |
| G1 | 01,02 | M1_texture | calibrated | 180 | 0.38 | 0.689 | 0.494 | 0.255 | 0.704 | 0.143 | platt |
| G1 | 01,02 | M2_dino | raw | 180 | 0.38 | 0.663 | 0.486 | 0.347 | 1.100 | 0.359 |  |
| G1 | 01,02 | M2_dino | calibrated | 180 | 0.38 | 0.663 | 0.486 | 0.259 | 0.713 | 0.193 | platt |
| G1 | 01,02 | M3_dino_distinct | raw | 180 | 0.38 | 0.659 | 0.484 | 0.350 | 1.109 | 0.363 |  |
| G1 | 01,02 | M3_dino_distinct | calibrated | 180 | 0.38 | 0.659 | 0.484 | 0.257 | 0.707 | 0.175 | platt |
| G2 (dev) | 03,04 | M0_constant | raw | 180 | 0.88 | 0.500 | 0.883 | 0.343 | 0.883 | 0.490 |  |
| G2 (dev) | 03,04 | M1_texture | raw | 180 | 0.88 | 0.635 | 0.924 | 0.308 | 0.810 | 0.448 |  |
| G2 (dev) | 03,04 | M1_texture | calibrated | 180 | 0.88 | 0.365 | 0.842 | 0.346 | 0.889 | 0.490 | platt |
| G2 (dev) | 03,04 | M2_dino | raw | 180 | 0.88 | 0.615 | 0.934 | 0.215 | 0.628 | 0.289 |  |
| G2 (dev) | 03,04 | M2_dino | calibrated | 180 | 0.88 | 0.385 | 0.876 | 0.372 | 0.944 | 0.517 | platt |
| G2 (dev) | 03,04 | M3_dino_distinct | raw | 180 | 0.88 | 0.619 | 0.935 | 0.212 | 0.620 | 0.285 |  |
| G2 (dev) | 03,04 | M3_dino_distinct | calibrated | 180 | 0.88 | 0.381 | 0.876 | 0.380 | 0.961 | 0.525 | platt |
| G3 | 05 | M0_constant | raw | 90 | 0.27 | 0.500 | 0.267 | 0.258 | 0.709 | 0.249 |  |
| G3 | 05 | M1_texture | raw | 90 | 0.27 | 0.593 | 0.312 | 0.210 | 0.603 | 0.108 |  |
| G3 | 05 | M1_texture | calibrated | 90 | 0.27 | 0.407 | 0.223 | 0.284 | 0.762 | 0.288 | platt |
| G3 | 05 | M2_dino | raw | 90 | 0.27 | 0.824 | 0.545 | 0.153 | 0.431 | 0.084 |  |
| G3 | 05 | M2_dino | calibrated | 90 | 0.27 | 0.176 | 0.174 | 0.270 | 0.733 | 0.259 | platt |
| G3 | 05 | M3_dino_distinct | raw | 90 | 0.27 | 0.780 | 0.432 | 0.172 | 0.479 | 0.155 |  |
| G3 | 05 | M3_dino_distinct | calibrated | 90 | 0.27 | 0.220 | 0.182 | 0.278 | 0.749 | 0.268 | platt |
| G4 | 06 | M0_constant | raw | 90 | 0.06 | 0.500 | 0.056 | 0.287 | 0.767 | 0.484 |  |
| G4 | 06 | M1_texture | raw | 90 | 0.06 | 0.579 | 0.100 | 0.193 | 0.574 | 0.361 |  |
| G4 | 06 | M1_texture | calibrated | 90 | 0.06 | 0.421 | 0.065 | 0.336 | 0.868 | 0.530 | platt |
| G4 | 06 | M2_dino | raw | 90 | 0.06 | 0.751 | 0.246 | 0.116 | 0.380 | 0.209 |  |
| G4 | 06 | M2_dino | calibrated | 90 | 0.06 | 0.249 | 0.045 | 0.290 | 0.774 | 0.487 | platt |
| G4 | 06 | M3_dino_distinct | raw | 90 | 0.06 | 0.751 | 0.237 | 0.121 | 0.393 | 0.218 |  |
| G4 | 06 | M3_dino_distinct | calibrated | 90 | 0.06 | 0.249 | 0.046 | 0.294 | 0.781 | 0.491 | platt |
| G5 | 10 | M0_constant | raw | 90 | 0.00 | — | — | 0.298 | 0.789 | 0.546 |  |
| G5 | 10 | M1_texture | raw | 90 | 0.00 | — | — | 0.407 | 1.032 | 0.630 |  |
| G5 | 10 | M1_texture | calibrated | 90 | 0.00 | — | — | 0.292 | 0.777 | 0.540 | platt |
| G5 | 10 | M2_dino | raw | 90 | 0.00 | — | — | 0.118 | 0.406 | 0.318 |  |
| G5 | 10 | M2_dino | calibrated | 90 | 0.00 | — | — | 0.278 | 0.750 | 0.527 | platt |
| G5 | 10 | M3_dino_distinct | raw | 90 | 0.00 | — | — | 0.124 | 0.419 | 0.325 |  |
| G5 | 10 | M3_dino_distinct | calibrated | 90 | 0.00 | — | — | 0.282 | 0.756 | 0.530 | platt |
| G6 | 08,09 | M0_constant | raw | 180 | 0.59 | 0.500 | 0.589 | 0.257 | 0.707 | 0.122 |  |
| G6 | 08,09 | M1_texture | raw | 180 | 0.59 | 0.514 | 0.634 | 0.297 | 0.856 | 0.235 |  |
| G6 | 08,09 | M1_texture | calibrated | 180 | 0.59 | 0.486 | 0.593 | 0.274 | 0.784 | 0.132 | platt |
| G6 | 08,09 | M2_dino | raw | 180 | 0.59 | 0.547 | 0.634 | 0.340 | 1.093 | 0.271 |  |
| G6 | 08,09 | M2_dino | calibrated | 180 | 0.59 | 0.547 | 0.634 | 0.256 | 0.705 | 0.113 | platt |
| G6 | 08,09 | M3_dino_distinct | raw | 180 | 0.59 | 0.542 | 0.630 | 0.341 | 1.096 | 0.271 |  |
| G6 | 08,09 | M3_dino_distinct | calibrated | 180 | 0.59 | 0.542 | 0.630 | 0.256 | 0.705 | 0.113 | platt |
| G7 | 11 | M0_constant | raw | 90 | 0.89 | 0.500 | 0.889 | 0.294 | 0.782 | 0.442 |  |
| G7 | 11 | M1_texture | raw | 90 | 0.89 | 0.304 | 0.860 | 0.501 | 1.355 | 0.589 |  |
| G7 | 11 | M1_texture | calibrated | 90 | 0.89 | 0.304 | 0.860 | 0.319 | 0.833 | 0.464 | platt |
| G7 | 11 | M2_dino | raw | 90 | 0.89 | 0.506 | 0.899 | 0.723 | 2.540 | 0.783 |  |
| G7 | 11 | M2_dino | calibrated | 90 | 0.89 | 0.506 | 0.899 | 0.568 | 1.519 | 0.678 | platt |
| G7 | 11 | M3_dino_distinct | raw | 90 | 0.89 | 0.484 | 0.892 | 0.731 | 2.610 | 0.787 |  |
| G7 | 11 | M3_dino_distinct | calibrated | 90 | 0.89 | 0.484 | 0.892 | 0.560 | 1.488 | 0.672 | platt |

AUROC and PR-AUC are undefined (—) for a held-out group with one outcome class. (dev) marks the fold containing localizer-dev region 03.

Reliability diagrams: `figures/stage6_8_pilot/reliability_<model>.png`. Per-row predictions and per-fold metrics: `results/stage8/pilot/`.
