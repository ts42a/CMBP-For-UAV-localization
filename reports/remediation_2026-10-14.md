# Inventory and readiness repairs — 2026-10-14

## Inventory and cleanup decision

The Excel workbook is `outputs/01a13a32/project-review/CMBL_inventory_and_fix_plan.xlsx`.
It lists individual project files and grouped bulk collections, their purpose,
size, references, KEEP/ARCHIVE/REVIEW/REMOVE CANDIDATE recommendation, and fix plan.
No files were deleted. In particular, the two untracked Word drafts, frozen
experiment evidence, original data, Git history and another chat's worktree remain.
Python bytecode is disposable; old generated builds/figures are archive candidates.
The sample dataset is a **review** candidate, not a verified duplicate.

## The twelve requested checks

| # | Requirement | Evidence / current result |
|---|---|---|
| 1 | H1–H3 numeric decisions, committed first | `configs/hypothesis_decisions_v1.yaml`, commit `b1af20f`. Prospective supplement before further evaluation; labels had already been generated/fetched. No claim that historical blinding can be restored. |
| 2 | Each Stage 11 kernel | `stage11_status_2026-10-14.json`: owned listing confirms 05/06/10 finished; 01/02/03/04/08/09/11 missing. Snapshot, not continuous monitoring. |
| 3 | R1–R6 and verification | Fail-closed contracts implemented; `verify.py` now includes integrity regression tests and every fetched full-label region. See details below. |
| 4 | Full grid and pilot overlap | 05: 4,257, 06: 3,096, 10: 1,296 unique required attempts; all 90 pilot outcomes per region match. Each has a new `integrity_check.json`. Empty overlap fails. |
| 5 | Stage 9 freeze | Commit `4cbc5e2`: selected hybrid with live fallback, fixed route/fidelity/bootstrap/stress seeds, synthetic-only `configs/drift_calibration.json`. Approval basis records the author's implementation instruction. Execution remains gated. |
| 6 | Model/runtime pins | `dinov2_artifacts_v1.json`: upstream revision and weight hash, all 157 Python source hashes. `runtime_inventory_2026-10-14.json`: PC Python 3.14.7 / torch 2.12.1+cpu; T4 Python 3.12.13 / torch 2.10.0+cu128. RTX 4080 unavailable, versions unmeasured. |
| 7 | Kaggle guide and tag | `kaggle/README.md` now documents Stage 11 and exact historical tag commit `88631e377a5ab8b006d9ad372ca516c1b77f8bdf`. It clearly distinguishes subsequent integrity repairs. |
| 8 | Replay and 2,000 live comparisons | `route_replay.py` core and synthetic tests implemented: drift, P0–P5, dynamic callback inputs, gating, scoring, matched-count comparator, fidelity thresholds. **Incomplete:** production adapter/model fitting, durable replay resume, P6 oracle, route uncertainty/stress orchestration, and actual 2,000-attempt live check. No policy-comparison result claimed. |
| 9 | Small PC baseline | Fresh 900-attempt pilot predictor batch reproduced the historical reference. `baseline_pc_2026-10-14.md`: texture AUROC 0.5524; DINO 0.6511; combined 0.6390. Pilot function check only. |
| 10 | Licence/data | MIT selected by author and added as `LICENSE`; `DATA_STATEMENT.md` records dataset and Google imagery boundaries and unresolved redistribution rights. |
| 11 | Methods/preregistration draft | Existing `CMBL_paper_revised_v0.2.docx` already contains methods, preregistration/deviation chronology and result placeholders. Preserved as the author's working draft. The new hypothesis supplement and implementation report need incorporation before submission. |
| 12 | 4080/T4 check then launch | `launch_gate.py` implemented, currently FAIL because actual RTX 4080 benchmark is absent. New Stage 11 dispatch is blocked until it passes. No new full-scale job launched. |

## Data-integrity fixes

* **R1:** exact expected frame/radius/offset/variant set, duplicate/schema/domain checks,
  protocol identities, expected pilot overlap and agreement, CUDA/TF32/commit evidence.
* **R2:** immutable experiment fingerprint covers configuration, selection, code and
  cache bytes. Resumes append events and cannot replace original provenance.
* **R3:** malformed/truncated CSV records fail; each frame writes through atomic
  replacement so interruption retains the previous complete file.
* **R4:** unknown/unlabelled outcomes, inconsistent acceptance, invalid coordinates,
  nonfinite quantities and mixed hashes are rejected before feature extraction.
* **R5:** ordered tile IDs/geometry, 384-dimensional finite embeddings, preprocessing,
  source rasters, DINO source and checkpoint bytes are checked. Eleven existing
  caches are explicitly registered. Registration establishes current byte identity;
  it is not retrospective proof of the original computation. Region 03's older
  single-raster metadata has a constrained, explicit compatibility path.
* **R6:** new feature NPZs store ordered attempt keys; consumers check file hashes,
  row order, dimensions, finite values and geographic membership. Only the exact
  historical pilot pair is grandfathered by recorded hashes; other unkeyed pairs
  must be regenerated.

Frozen matcher, preregistration, splits and historical prediction files were not
rewritten. Two full verifier runs were requested; individual logs/results live in
`build/verify/`, with the latest summary in `build/verify/last_verify.json`.
Both full runs passed all nine checks; the second ran 13 regression tests and
checked all three fetched regions (`build/verify/20261014_231124`). Golden label
outcomes were identical with zero position difference on 18 attempts. The separate
run in `build/verify/20261014_231448` also verified source-raster/cache identities
for 05/06/10. This is separate from the required 180-attempt RTX 4080 comparison.
All modified/new Python sources parse; Git whitespace checks pass with CRLF-aware
settings, preserving the repository's existing file line endings.

## Additional review items

Safe ZIP staging/promotion and traversal checks replace source-overwriting result
imports; TAR extraction uses its data filter. Original imported results are backed
up. Semantic manifest identity permits image-path relocation without changing the
frozen split bytes. Unsupported predictor protocol versions fail. Input NPZ,
split, feature-manifest, quality and AGL identities are recorded. Reports distinguish
the registered historical pilot pair from other inputs. Supplementary fixed-seed
paired group contrasts are produced separately from golden summaries.

Performance refactoring (R11) is deferred until profiling shows a useful bottleneck.
R10 is partial: unknown protocol versions are rejected, but complete typed schema
validation and explicit defaults throughout historical estimators are not finished.
R13 is the incomplete replay integration above. R14 has a paired-bootstrap utility;
full H1 real-vs-synthetic matched-fold evidence is still unavailable.

## Remaining execution order

1. Finish replay production adapter, durable checkpoints, oracle and inference
   orchestration against synthetic/pilot fixtures; keep these separate from outcomes.
2. The author confirmed the RTX 4080 will be a cloud computer. Use
   `kaggle/CLOUD_4080.md` and `cloud_4080_benchmark.py` once that host is available,
   run the fixed 180 attempts and record its environment.
   The new launch gate must pass before any new full-scale job is pushed.
3. Package a committed repair revision (do not reuse the historical tag to imply it
   contains repairs), launch the seven missing regions, and revalidate all ten.
4. Build keyed full features, fit only the appropriate source folds, collect the
   predeclared 2,000 live replay comparisons and select hybrid or live from that result.
5. Run the frozen hypotheses and replay analyses, then replace paper placeholders.

No full-scale hypothesis or replay claim is supported merely by this engineering
verification. The project is **not ready to launch under the requested 4080 gate**.
