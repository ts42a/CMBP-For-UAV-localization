# History rewrite before publishing (2026-09-30)

Before the first push, the history was rewritten with git-filter-repo: (1) UAV-VisLoc-derived files
(per-frame metadata, outcome tables, figures showing dataset imagery, Kaggle account paths) were removed
from every commit, and two hard-coded map coordinates were replaced; (2) author, committer and tagger
emails were mapped to the GitHub no-reply address; (3) co-author trailers were removed from commit messages. File contents are otherwise unchanged, but every
commit ID changed. Reports, run manifests and Kaggle outputs written before the rewrite cite the
original IDs (for example `bundle_commit` in env.json, `git_commit` in run_manifest.json); this table
maps them. The Stage 11 code bundle `25b6060` is tag `stage11-labels-v1`.

| original | now | date | subject |
|---|---|---|---|
| `1f08dd4` | `6b2bac4` | 2026-09-27 | Baseline: Stages 1-2 audits and Stage 3B/3C matcher freeze |
| `a4ab63a` | `d4ea2bd` | 2026-09-27 | Freeze Stage 3 splits and pre-registration; add windowed labeller |
| `e53b1c9` | `cdb33e5` | 2026-09-27 | Stage 4 checks: mechanics pass, zero-offset verification fails |
| `2117406` | `1513693` | 2026-09-27 | Declare verifier-switch criteria before any patch-matcher run |
| `ecb4fc9` | `9ab4817` | 2026-09-27 | Add DINOv2 patch-token verifier candidates (not frozen) |
| `57b9df2` | `acd2845` | 2026-09-28 | Re-freeze localizer with DINOv2 patch verifier (deviation D1) |
| `7c1edc9` | `d15eb4a` | 2026-09-28 | Val gate and Stage 4 checks pass with the re-frozen patch verifier |
| `e2bd42a` | `9e37737` | 2026-09-28 | Pre-declare Stage 5 go protocol; fix unregistered gate; deprecate superseded scripts |
| `5091154` | `448ebba` | 2026-09-28 | Disclose partial pilot data seen before the Stage 5 go protocol was written |
| `23a9e2d` | `1327f9a` | 2026-09-28 | Stage 5 multi-region pilot: gate GO, three issues flagged for decision |
| `28fb7e6` | `1eda635` | 2026-09-28 | Freeze Stage 6-8 predictor protocol and deviation D2 before any predictor run |
| `2d34e3d` | `5c8b5dc` | 2026-09-28 | Audit along-track ground-truth offsets on the Stage 5 pilot (read-only) |
| `e66ed28` | `6e4bd2e` | 2026-09-28 | Stages 6-8 pilot function check: map features, predictors, calibration |
| `b2bfdd6` | `a0f0e08` | 2026-09-28 | Predictor protocol v2 (deviation D3): fixed-effects Platt calibration |
| `bbc0b7c` | `853758c` | 2026-09-28 | Stages 6-8 pilot under protocol v2: calibration no longer inverts |
| `f76a0ab` | `c139c2b` | 2026-09-28 | Predictor protocol v3 (deviation D4): global-permutation leakage control |
| `b55865a` | `552d418` | 2026-09-28 | Stages 6-8 pilot under protocol v3: leakage control passes (0.506) |
| `428c362` | `6fe328e` | 2026-09-28 | Stage 10 kit: frozen benchmark rules, Kaggle runner, portability fix, FP16 check, cost sheet |
| `563cf1d` | `372a46b` | 2026-09-28 | Stage 10 simulated Kaggle run passes; record status |
| `afa1fee` | `1eb4c16` | 2026-09-29 | Stage 10: Kaggle API automation for upload, GPU run and fetch |
| `93bfed2` | `40ff147` | 2026-09-29 | Stage 10: record Kaggle quota; fix dataset-exists check before first upload |
| `7ae60df` | `a57ef59` | 2026-09-29 | Predictor protocol v4 (deviation D5): exclude pilot frames from headline metrics |
| `0e13810` | `e84c10a` | 2026-09-29 | Engineering hygiene: verify command, pinned environment, artefact archiving |
| `4768684` | `db32112` | 2026-09-29 | Height above ground, frame quality, protocol v5 (proposed D6): flight diagnostics |
| `5586197` | `6c14971` | 2026-09-29 | H1 generator and Stage 9 protocol (both proposed); measured localizer reach |
| `c2275af` | `cd60b52` | 2026-09-29 | Prepare repository for GitHub: README, ignore rules, local-only files untracked |
| `473ace0` | `c885f67` | 2026-09-30 | Kaggle kernel: writable torch hub; package local manifests; verify SKIP status |
| `3daa44d` | `4367096` | 2026-09-30 | Stage 10 GO on Kaggle GPU; D6 approved; Stage 11 full-label run frozen |
| `25b6060` | `88631e3` | 2026-09-30 | package_kaggle: resolve --ref to its commit (annotated tags) |
| `c5eb086` | `013650c` | 2026-09-30 | Kaggle kernel: stop when the mounted code bundle is not the staged job's commit |
