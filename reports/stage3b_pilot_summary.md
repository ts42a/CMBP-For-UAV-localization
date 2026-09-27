# Stage 3B — Pilot / recovery note

**Canonical decision:** see **`reports/stage3b_decision.md`** → **READY**  
**Frozen matcher:** `dinov2_c2f` in `configs/matcher_stage3b_frozen.yaml`

An earlier draft pilot under `results/pilot/` (same region, no pre-frozen val split) is **superseded** by this recovery protocol: splits frozen before tuning, visual credibility required on untouched val, Stage 3A baseline preserved.

Key val evidence (`dinov2_c2f`): **9/30** success@100 m and **21/30** failures; ≥5 human-verified credible successes; Stage 3A baseline remains **0/100** @100 m on the comparison set.
