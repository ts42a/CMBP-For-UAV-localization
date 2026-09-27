# Gap Decision — Stage 1

**Date:** 2026-09-22  
**Decision:** **GO**  
**Bound:** Based on reviewed full texts (RSPNet, Jiang 2024, Robust Belief Roadmap, SafeVPR, SASGeo), SciOpen/OpenAlex metadata for MLSNet (full text unavailable), and one focused localizability search. This is **not** proof that no paper outside the search combines the four criteria.

---

## Four criteria

| # | Criterion | Required sense for “overlap” |
|---|---|---|
| 1 | Real UAV localization outcomes | Labels/evaluation from real UAV geo-localization results (not only simulated queries, hang-flight-as-aspiration, or ground-robot VPR) |
| 2 | Held-out independent regions | Geographic generalization protocol with held-out regions (not merely distinct images or scene categories) |
| 3 | Measured probability calibration | Explicit calibration quality of predicted success/risk probabilities (e.g. reliability/ECE/Brier or equivalent), not only ranking/AUC/accuracy |
| 4 | Evaluated scheduling using numeric probabilities | Policy that uses numeric probabilities to schedule/select localization attempts, with measured maintenance/navigation outcomes |

---

## Criterion matrix (reviewed evidence)

| Paper | (1) Real UAV outcomes | (2) Held-out independent regions | (3) Probability calibration | (4) Numeric-probability scheduling |
|---|---|---|---|---|
| RSPNet | No — simulated \(L_T\) labels/eval (§3.1) | Partial/author-stated region split (§4.1); independence not independently verified | Not reported in reviewed material | Not evaluated (planning only suggested) |
| Jiang 2024 | No — simulated queries for quantitative matching (§4.1–4.3) | Not reported (category splits) | Not reported | Not evaluated |
| MLSNet | **Unknown** — full text unavailable; abstract says remote-sensing matchability segmentation | **Unknown** from reviewed material | **Unknown** from reviewed material — **do not claim absent** | **Unknown** from reviewed material |
| Robust Belief Roadmap | No — computational belief planning | N/A (not an image-region study) | No (assumed \(p_j\), not calibrated predictor) | Partial — uses numeric \(p_j\) for path choice, but \(p_j\) are known sensing rates, not map-predicted UAV success |
| SafeVPR | No — ground-robot sequence VPR | Cross-condition, not UAV geographic holdouts | Yes — conformal FDR validity | No — accept/reject after retrieval, not intermittent map-correction scheduling |
| SASGeo | No — synthetic PoC; not real flight | Future work only | Design mentions calibration; PoC uses **uncalibrated** margin | Design has accept costs; not real-route probability scheduling |
| Search: 2407.08162 VPR integrity+nav | No UAV map outcomes (ground robot) | Partial spatial separation Campus/Office | Integrity classifier, not ECE of map success probs | Navigation accept/reject — not map-probability scheduling |
| Search: VPR uncertainty / LiDAR localizability | No / wrong modality | Not the UAV map protocol | Uncertainty ≠ map-success ECE | Not UAV correction scheduling |

**No reviewed paper is evidenced to combine (1)+(2)+(3)+(4).**

---

## Why not NARROW

NARROW would require at least one prior that already combines all four. The closest pieces each miss multiple criteria:
- SafeVPR: strong on (3)-like risk control and (decision evaluation), weak on (1)(2)(4) as defined above.
- RBRM: strong on intermittent-sensing planning with numeric probabilities, weak on (1)(2)(3) as map-learned UAV success prediction.
- RSPNet/Jiang: closest on map matchability prediction, weak on (1)(3)(4); (2) only partially for RSPNet.

---

## Why not UNRESOLVED

MLSNet full text is missing, which is a **residual verification risk**, especially for criteria (2) and (3). Nonetheless a defensible GO remains because:
1. Fully readable matchability priors (RSPNet, Jiang) do not combine the four criteria.
2. Fully readable decision/calibration priors (SafeVPR, RBRM, SASGeo, search hits) do not supply real UAV map-outcome + geographic holdout + map-probability scheduling together.
3. MLSNet’s **available abstract** frames a reference-image matchability **segmentation** task on remote sensing imagery — the same family as RSPNet/Jiang — without stating real UAV localization outcomes or numeric-probability scheduling. Even if unread sections added calibration or geographic splits, criteria (1) and (4) are not supported by the abstract framing; they remain unverified rather than affirmed.

If MLSNet full text later shows real UAV outcomes **and** held-out regions **and** probability calibration **and** evaluated probability scheduling, revisit and possibly change to NARROW.

---

## Precise gap (bounded)

Within the reviewed set, the open combination is:

> **Map-feature prediction of a frozen UAV visual localizer’s success on real UAV imagery in geographically held-out regions, with measured probability calibration, and evaluation of intermittent correction scheduling that consumes those numeric probabilities.**

This extends (does not invent) matchability prediction (RSPNet/Jiang family) and intermittent-sensing planning (RBRM family) / post-hoc VPR integrity (SafeVPR family), under a different information setting and outcome definition.

---

## Strongest potentially overlapping paper

**Among named Stage-1 papers:** **SafeVPR (arXiv:2605.28048)** — strongest overlap on calibrated risk control and evaluated accept/reject decisions; still a different setting (post-observation sequence VPR on ground-robot datasets).

**Among search hits:** **“Improving Visual Place Recognition Based Robot Navigation Through Verification of Localization Estimates” (arXiv:2407.08162)** — strongest overlap on closing the loop from predicted match integrity to real navigation decisions; still not UAV–satellite map-only success prediction with geographic calibration and probability scheduling.

---

## Evidence still needed

1. **MLSNet full PDF** (methods, experiments, supplementary) — verify or refute proposal claims on modality/split/metrics; check for calibration and geographic protocols.
2. Public path for **WHU-SCENE-MATCHING** / RSPNet code (or confirm permanently unavailable).
3. Stage 2: UAV-VisLoc **6,742 vs Zenodo 14,998** identity audit (do not mix releases; no download in this stage).
4. Optional deepening of search (additional queries / citation chaining from SafeVPR and 2407.08162) if a reviewer challenges search completeness.

---

## If novelty were judged insufficient

Fallback framing (only if later evidence collapses the gap): **replication / stress-test** of map matchability predictors (RSPNet-inspired) under **real UAV-VisLoc outcomes**, with **mandatory geographic holdouts**, **calibration metrics**, and **paired recorded-route scheduling** — reporting transfer and correlated-failure analysis even if ranking gains are modest.
