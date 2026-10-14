# Local cleanup and workflow — 2026-10-14

## Cleanup completed

Removed 97 regenerable project `.pyc` files (1,271,399 bytes). Verified all 770
files in `datasets/sample/03` against the retained full region-03 dataset using
SHA-256, updated the sole preview-script dependency, and removed that duplicate
directory (3,428,281,295 bytes, about 3.19 GiB). No original full data, experiment
results, paper drafts, virtual environment or other chat's worktree was deleted.
Audit: `reports/cleanup_2026-10-14.json` and
`build/cleanup/sample03_duplicate_check.json`.

## Local implementation

* Root `test.py`: interactive Local/Full menu, data percentage, CPU threads,
  deterministic proportional contiguous selection, plan preview, status, stop,
  process-tree termination, exclusive writer lock and resumable stage hashes.
* Both real and synthetic label streams use the identical selected frames and
  full radius/offset grid. Labelled outputs are checked against that exact grid.
* Executable typed predictor settings preserve the historical defaults. Source-only
  reusable predictor fits have numerical parity tests and reject held-out leakage.
* Shared, bounded map feature service supports arbitrary replay estimates. All 900
  historical pilot feature rows and both embedding arrays reproduced exactly.
* H1 now trains real/synthetic models on matched source rows and evaluates both
  on real held-out labels. H1–H3 decisions apply the committed rules; local runs
  are explicitly exploratory even if the numerical criteria are met.
* Label-noise upper-bound derivatives now carry correct producer hashes and
  ordered keys, rather than copying a manifest that describes different bytes.
* Real live replay supports P0–P6, source-route threshold streams, accepted-fix
  gating, exact-centre persistent live caches, matched equal-spacing controls,
  matched P4/P2/P3 caps, failure blocks, failure-run diagnostics and paired route
  block intervals. Full mode schedules all 80 level/budget/seed cells and combines
  the five paired seeds for the primary H4 interval.
* A fidelity report validates recorded attempts against the live cache before
  comparison. It requires 2,000 distinct live attempts and coverage of all levels,
  budgets and supplied routes. A small local run may correctly report INCOMPLETE;
  live source A remains in use. No hybrid pass is fabricated.

Operational details are fixed in `configs/replay_execution_v1.json`, including
source P0 reference streams for thresholds and exact-centre caching. These clarify
the implementation before the new local replay; they are not silently retrofitted
to any historical result. Threshold ties and realised source/target rates are
reported rather than assumed to match nominal budgets.

## Validation evidence

21 regression tests pass. A live four-frame/two-region replay integration check
completed, as did one synthetic frame with all nine attempts. The label-noise
pipeline completed both registered and upper-bound pilot evaluations. A small
real-map profile measured a 1.76 s cold feature query and approximately 0.00027 s
per repeated warm query, with exact output parity; this is not a claimed whole-run
speedup (`reports/map_feature_profile_2026-10-14.json`).

The four-thread full verifier passed label equivalence (18 identical outcomes,
zero position difference) but failed the three **byte-for-byte** predictor checks
because predictions differed below 1e-10. The original thread environment reproduced
v3 byte-for-byte. A fresh full verifier in the original environment was also
started; its evidence is under `build/verify/`. Do not describe the four-thread
byte comparison as a pass. Thread settings are part of the new run identity.

## Requested 8% execution

Command:

```cmd
python test.py --mode local --percent 8 --threads 4
```

Selection: 540/6,744 frames (8.007% after integer allocation), ten regions excluding
07, 4,860 real and 4,860 synthetic attempts. Local replay uses one L2 / 10% budget /
seed 91001 cell with all policies and stress checks. This is an engineering/development
run, not a replacement for the registered full-scale experiment.

Output: `results/local/run_8pct_seed20261014/`. `status.json` is the live execution
record; `logs/` holds each stage's output; `REPORT.md` is written when the workflow
finishes. Execution may take many CPU hours. A running job is not a completed result.

```cmd
python test.py --status
python test.py --stop
```

Repeat the original run command to resume. Keep the computer awake. Do not change
code or protocols during the run; a changed experiment requires a new output folder.

## Deferred

Actual cloud RTX 4080 environment/180-attempt equivalence, new production Stage 11
launches, and full-data confirmatory results remain deferred. The Full menu option
checks the actual 4080 acceptance gate before execution. No cloud instance or new
Kaggle job was started by this local work.
