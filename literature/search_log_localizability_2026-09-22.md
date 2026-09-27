# Search log — localizability prediction (VPR / LiDAR)

**Date:** 2026-09-22  
**Purpose:** One focused pass for work that could invalidate the proposed four-criterion gap.  
**Bound:** Not exhaustive; queries and inclusion decisions recorded below.

## Queries

1. `localizability prediction visual place recognition OR LiDAR map matchability calibration`
2. `"localizability" OR "matchability prediction" OR "localization success prediction" UAV OR "visual place recognition" satellite map`
3. `"probability calibration" OR ECE OR "reliability diagram" "visual place recognition" OR "UAV visual localization" map features`

## Candidates and decisions

| Candidate | Source hit | Decision | Reason |
|---|---|---|---|
| Nubert et al., Learning-based Localizability Estimation… IROS 2022 | Q1 | Read extract; **adjacent** | LiDAR scan localizability; wrong modality for optical UAV–map success |
| Towards Robust LiDAR Localization… arXiv:2509.18954 | Q1 | Skim extract; **exclude core** | ICP covariance prediction; LiDAR |
| Neural Error Covariance… arXiv:2501.02558 | Q1 | Skim; **exclude core** | LiDAR map-matching covariance |
| SOLVR arXiv:2409.10247 | Q1 | Skim; **exclude** | LiDAR–visual relocalisation; not map-success probability scheduling |
| Laskar et al., On Estimation of Image-matching Uncertainty in VPR, arXiv:2404.00546 | Q2 | **Read**; save `sources/search_VPR_uncertainty_2404.00546.txt` | Strong VPR uncertainty baseline; lacks UAV map holdout + scheduling combo |
| VPR integrity → navigation, arXiv:2407.08162 | Q2 | **Read**; save `sources/search_VPR_verification_nav_2407.08162.txt` | Strongest search overlap on decisions; ground robot |
| Sequence-Matching Receptiveness Prediction arXiv:2503.06840 | Q2 | Note only; **adjacent** | Predict when sequence matching helps; not map-only UAV success probs |
| NaviLoc / FoundLoc UAV papers | Q2 | Note; **adjacent** | UAV localization systems; not the calibrated map-success + scheduling study |
| Gronat et al. CVPR 2013 per-location VPR calibration | Q3 | Skim extract; **adjacent** | Score calibration across places; different task |
| Classifier calibration surveys | Q3 | Background only | Not domain priors |

## Outcome for gap

No search candidate, as reviewed, combines real UAV localization outcomes + held-out independent regions + measured probability calibration of map-based success probabilities + evaluated numeric-probability scheduling of localization updates.

Saved extracts under `sources/search_*.txt`.
