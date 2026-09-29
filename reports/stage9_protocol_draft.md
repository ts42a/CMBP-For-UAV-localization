# Stage 9 replay protocol — draft for approval

Status: **proposed** (`configs/stage9_protocol_v1.yaml`). Nothing has been replayed. Approve, change or reject each item marked **Decide**; once approved the YAML status becomes `frozen_before_replay`.

## What the replay answers
RQ3 / H4: along the recorded flights, does attempting localization where the calibrated predictor expects success keep the UAV localized better than equal spacing, at the same number of actual attempts? Maintenance only (fraction of the route within 100 m, loss-of-localization episodes), not A-to-B arrival.

## Design choices driven by the pilot evidence
| Evidence (project evaluation 2026-09-29) | Choice |
|---|---|
| Within-frame (radius/offset) variation is 24% of pilot label variance; 64% of frames give the same outcome on all 9 attempts | Outcome lookup from the full-scale label table is expected to be faithful → option C with a pre-declared fidelity check |
| Effective reach ≈ R + 250 m (dev region 03 at R = 100 m: 100% at 0 m offset, 17% at 250 m, 0% from 500 m) | Attempt radius R ≥ 2σ keeps the truth within reach while σ ≤ 500 m; fixes gated at max(3σ, R + 300 m) |
| Accepted-fix error median 26.5 m, p75 41 m | σ after a fix = 30 m |
| Pre-registration uses prior σ = R/2 | Attempt radius = smallest pre-registered R ≥ 2σ |
| Routes 9–103 km, 60–130 m between frames | Levels 2%, 5%, 10% of distance: ~60–100 m, ~150–250 m, ~300–500 m of drift between typical fixes |

Effective reach measured on the dev region only: `reports/stage9_reach_r03.md`.

## Decide
1. **Outcome source (the cost lever).** A live ≈ 622 CPU-h (placeholder design); C hybrid ≈ a few CPU-h (2,000-attempt live fidelity check plus live runs only when drift exceeds the labelled offsets). Recommended: **C**, used only if the fidelity check passes (≥ 90% identical outcomes, ≥ 90% of accepted positions within 25 m), otherwise A.
2. **Drift levels** L1/L2/L3 = 2% / 5% / 10% of distance travelled (plus L0 = 0 as a sanity check). Real visual-inertial odometry drift is typically ~1–5%; 10% is a stress level.
3. **Primary H4 cell**: P4 calibrated vs P0 equal spacing at budget 10%, level L2. Every other cell is reported, not tested.
4. **Untuned product rule** τ = 0.9 (attempt once the skipped frames' combined chance of at least one success reaches 90%).
5. **Gate** = max(3σ, R + 300 m). A real filter would reject a fix far from its estimate; without a gate one accepted-incorrect fix can derail a route.

## Not decided here
- Destination radius (not needed for maintenance metrics; pre-registration defers it).
- Closed-loop A-to-B simulation (RQ4, optional).

## Cost
With C: replay compute is dominated by map features at the estimated positions (cached tile embeddings; milliseconds per attempt) plus the ~3 CPU-h fidelity check. With A: the Stage 10 GPU benchmark decides whether it fits in three weekly Kaggle quotas.
