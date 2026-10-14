# Stage 10 on Kaggle — step by step

Frozen rules: `configs/stage10_benchmark.yaml`. The benchmark runs the frozen localizer on 20
frames (2 per region, region 07 excluded) = 180 attempts, all of which already have PC results
in `results/stage5/pilot/outcomes.csv`, so the two can be compared directly.

## Automated route (Kaggle API) — does steps 2-5 from the PC

One-time: `py -3 -m pip install --user kaggle`, then create an API token at
https://www.kaggle.com/settings/api and save it as `~/.kaggle/access_token` (or `kaggle.json`).

```
py -3 scripts/protocol/package_kaggle.py              # step 1 (bundles)
py -3 scripts/protocol/kaggle_stage10.py check        # username + weekly GPU quota -> reports/stage10_kaggle_quota.json
py -3 scripts/protocol/kaggle_stage10.py stage        # upload folders (junctions, nothing copied) + kernel metadata
py -3 scripts/protocol/kaggle_stage10.py upload       # 3 PRIVATE datasets; timed -> reports/stage10_kaggle_transfer.json
py -3 scripts/protocol/kaggle_stage10.py push         # GPU (T4) script kernel, internet off: kaggle/stage10_kernel.py
py -3 scripts/protocol/kaggle_stage10.py wait         # poll until it finishes
py -3 scripts/protocol/kaggle_stage10.py fetch        # download, unpack to results/stage10/kaggle_bench, compare, cost sheet
```

`kaggle/stage10_kernel.py` finds the inputs by content (extracted folders or unextracted
archives), copies the code to `/kaggle/working/CMBL`, runs `stage10_runner.py` unchanged and
writes `/kaggle/working/stage10_kaggle_bench.zip`. It stops before running anything if a
Python module, an input or a benchmark region is missing, and writes `kernel_discovery.json`
saying which. The manual route below does the same by hand.

## 1. On the PC: build the upload bundles

```
py -3 scripts/protocol/package_kaggle.py
```

This writes to `build/kaggle/` (not committed):

| file | contents | size |
|---|---|---|
| `cmbl-code-<commit>.zip` | committed code: `scripts/`, `configs/`, `datasets/manifests/`, `kaggle/` | < 1 MB |
| `cmbl-cache.zip` | `embeddings/` (tile-embedding caches) + `torch_home/hub/` (DINOv2 repo and weights) | ~110 MB |
| `bundle_manifest.json` | commit and sha256 of both zips | |

## 2. On kaggle.com: create three **private** datasets

The dataset README states no licence, so keep every upload **private** (personal compute only).

| Kaggle dataset (suggested slug) | upload | note |
|---|---|---|
| `uav-visloc` | the folder `datasets/raw/UAV_VisLoc_dataset/` (region folders `01`…`11`, ~24 GB: 8.7 GB JPG + 15.3 GB uncompressed TIF) | time the upload: this is the data-transfer measurement. Keep TIFs uncompressed (they are memory-mapped) |
| `cmbl-cache` | `cmbl-cache.zip` | Kaggle unzips it |
| `cmbl-code` | `cmbl-code-<commit>.zip` | new version whenever the code changes |

Check the dataset size limit shown in the upload dialog; if 24 GB is not allowed, stop: the upload has to be split (e.g. one dataset per region group) before the kernel can run.

## 3. Notebook settings

- Accelerator: **GPU** (T4 or P100). Internet can stay **off** (weights come from `cmbl-cache`).
- Add the three datasets as inputs. Note their mount paths under `/kaggle/input/` (they depend on the slugs, and on whether Kaggle kept the zip's top-level folder).
- Before starting, note your weekly GPU quota and the session cap shown on your account.

## 4. Notebook cells

```
!cp -r /kaggle/input/cmbl-code/CMBL /kaggle/working/CMBL
!pip list 2>/dev/null | grep -iE "^(torch|torchvision|opencv|tifffile|pyyaml|numpy) "
```

If `tifffile` is missing: `!pip install tifffile` (needs internet; otherwise add it as a wheel dataset).

```
!python /kaggle/working/CMBL/kaggle/stage10_kernel.py
```

The kernel finds the three inputs by content, gives `torch.hub` a writable hub folder (it writes
`trusted_list` there, and `/kaggle/input` is read-only: passing `--torch-home /kaggle/input/...`
straight to `stage10_runner.py` fails with `PermissionError`), runs the runner and zips the results.

Expected: about 20 frames at GPU speed plus a short region-10 embedding benchmark. If the session
stops, rerun the same cell; the labeller resumes without duplicates.

Download `stage10_kaggle_bench.zip` from the notebook's output panel.

## 5. Back on the PC

Use `py -3 scripts/protocol/kaggle_stage10.py fetch` to stage and validate the
download before promotion. Existing results are retained as a timestamped backup.
For already imported results:

```
py -3 scripts/protocol/stage10_compare.py results/stage10/kaggle_bench --name stage10_benchmark
py -3 scripts/protocol/cost_projection.py --timing results/stage10/kaggle_bench/outcomes.csv \
    --label kaggle_gpu --runner-timing results/stage10/kaggle_bench/timing.json --weekly-quota-h <your quota>
```

`stage10_compare.py` applies the frozen agreement criteria (≥ 97% identical outcomes, accepted
positions within 5 m for ≥ 95%) and lists every disagreement. The cost sheet then decides whether
to stay on Kaggle (projection ≤ 3 × weekly quota).

## 6. Stage 11 full labels

The historical `stage11-labels-v1` tag resolves to the exact commit
`88631e377a5ab8b006d9ad372ca516c1b77f8bdf`. Downloaded jobs record its pre-history-rewrite
identity `25b60606259d4937632407d0629b613ae6388996`; the documented mapping is in
`reports/history_rewrite_2026-09-30.md`. These are equivalent historical identities.
The integrity repairs made on 2026-10-14 are subsequent changes; that historical
tag does **not** contain the repairs. Do not move the tag or claim that it does.

1. Run `.venv\Scripts\python.exe scripts/protocol/verify.py` on the PC. It checks
   every fetched full-label region, model/cache identity, and the golden pilot.
2. Run `py -3 scripts/protocol/stage11_status.py` for a live, timestamped Kaggle
   status snapshot. On 2026-10-14, 05/06/10 finished and were fetched and accepted;
   01/02/03/04/08/09/11 were absent from the owned-kernel listing. Region 07 is excluded.
3. Before a new launch, run the Stage 10 runner on the actual RTX 4080 with
   `--out results/stage10/rtx4080_bench`. Use the same 20 frames / 180 attempts,
   float32, TF32 off, and verified DINOv2 artifacts. Record its real runtime;
   CPU and T4 runtime versions cannot stand in for the 4080.
4. Run `.venv\Scripts\python.exe scripts/protocol/launch_gate.py`. It requires
   all 180 keys, identical outcome labels against the T4, and the registered
   accepted-position agreement. Missing evidence fails closed. Stage 11 dispatch
   now calls this gate before pushing new jobs.
5. Commit and package the chosen source revision with `package_kaggle.py --ref
   <exact-commit>`, then stage/upload the code and cache using the existing commands
   above. Record that new revision separately from the historical tag. A new
   bundle must contain `dinov2_artifacts_v1.json` and `legacy_artifacts_v1.json`.
6. After the launch gate passes, `py -3 scripts/protocol/kaggle_stage10.py labels
   --regions 01,02,03,04,08,09,11` handles jobs and fetches each result. The planned
   total is 6,744 frames / 60,696 attempts across ten regions. The acceptance
   check requires every frame × {100,300,1000} m × {0,1,2} offset, unique keys,
   valid labels/provenance, and all expected pilot overlap. Zero overlap fails.
7. Keep each `integrity_check.json`, `env.json`, `run_manifest.json` and original
   outcome CSV. Fetching stages ZIP contents under `build/kaggle/staging`, validates
   before promotion, and retains any replaced result. Never extract an output ZIP
   over source code. For an explicitly chosen new commit, use
   `stage11_acceptance.py --expected-commit <exact-commit> --regions <regions>`.

Legacy label runs have no immutable resume fingerprint. Keep them as historical
evidence and use a new output directory with the repaired runner. New resumes
require unchanged protocol, frame selection, code and cache identities.

The DINOv2 pin is `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; weights SHA-256 is
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
Actual observed runtimes are in `reports/runtime_inventory_2026-10-14.json`.
The 4080 benchmark and 2,000-attempt replay fidelity check remain outstanding.
