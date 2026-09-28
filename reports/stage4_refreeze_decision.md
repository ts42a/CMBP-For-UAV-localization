# Localizer re-freeze (deviation D1) — decision record

**Status: FROZEN and GATED** — `configs/matcher_stage4_frozen.yaml` (`dinov2_patch_s23`, accept rule `patch_20`).
The config file itself still says `gate: FROZEN_PENDING_GATES`; it is left unedited so its
sha256 matches the hash recorded in every gate and label output. This file records the gate results.

| Step | Data | Result | Evidence |
|---|---|---|---|
| Criteria declared | — | before any patch run | `reports/verifier_switch_protocol.md` (commit 2117406) |
| Verifier comparison | region 03 tune, 216 windowed + 24 zero-offset attempts | **IMPROVES** — 188 vs 7 accepted-correct, 0 incorrect, 24/24 frames, 24/24 zero-offset, 46/46 visual | `reports/verifier_comparison.md`, `reports/verifier_visual_review.csv` |
| Re-freeze | — | commit 57b9df2 | `configs/preregistration_v1.yaml` deviation D1 |
| Stage 3B val gate rerun | region 03 val, 30 untouched frames, full-map search | **PASS** — 17 accepted-correct, 1 accepted-incorrect, 12 rejected; precision 0.94; 17/17 visual | `reports/stage3b_gate_v2.md`, `reports/stage3b_gate_v2_visual_review.csv` |
| Stage 4 checks rerun | region 03 tune (5 frames) | **PASS** — zero-offset 5/5, determinism, 2 km control 0 correct, kill/resume identical | `reports/stage4_checks.md` (AKAZE run kept as `reports/stage4_checks_v1_akaze.md`) |

## Known limits carried forward
- Only failure seen on val: `03_0271`, a repetitive village-strip scene matched to a similar strip
  1.25 km away at exactly 20 inliers. Self-similar terrain is the expected failure mode.
- All evidence so far is region 03, the localizer-dev region; nothing here is a held-out result.
- Cost: ~6.6 s wall per windowed attempt on CPU (1.7 s at R = 100 m, ~13 s at R = 300/1000 m), ~6x AKAZE.
- Panel display bug fixed during review (query rotation used bank angle); matcher outputs unaffected.
