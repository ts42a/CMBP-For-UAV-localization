# Localization scripts (Stages 3A–3B)

## Stage 3A (baseline; preserved)

```bash
python scripts/localization/run_stage3a.py
python scripts/localization/stage3a_inspect_figures.py
```

Frozen: `pyramid_ncc_rot4` — **NOT READY** for labels (`reports/stage3a_pilot_summary.md`).

## Stage 3B (matcher recovery)

```bash
python scripts/localization/freeze_stage3b_splits.py
python scripts/run_stage3b_recovery.py
python scripts/localization/finalize_stage3b_decision.py
```

Frozen: `dinov2_c2f` — **READY** for full real-outcome labeling per `reports/stage3b_decision.md`.

Modules: `dino_embedder.py`, `retrieval.py`, `fine_verify.py`, `tile_index.py`, `geo_utils.py`, `stage3b_diagnose.py`, `freeze_stage3b_splits.py`.

