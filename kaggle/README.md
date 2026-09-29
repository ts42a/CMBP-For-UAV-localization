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
!python /kaggle/working/CMBL/kaggle/stage10_runner.py \
    --code-root /kaggle/working/CMBL \
    --data-root /kaggle/input/uav-visloc/UAV_VisLoc_dataset \
    --emb-root /kaggle/input/cmbl-cache/embeddings \
    --torch-home /kaggle/input/cmbl-cache/torch_home
```

Adjust the three `/kaggle/input/...` paths to what the input panel shows. `--data-root` must be the
folder that directly contains `01`, `02`, … `11`.

Expected: about 20 frames at GPU speed plus a short region-10 embedding benchmark. If the session
stops, rerun the same cell; the labeller resumes without duplicates.

```
!cd /kaggle/working/CMBL && zip -r /kaggle/working/stage10_kaggle_bench.zip results/stage10/kaggle_bench
```

Download `stage10_kaggle_bench.zip` from the notebook's output panel.

## 5. Back on the PC

Unzip into the repo so the files land in `results/stage10/kaggle_bench/`, then:

```
py -3 scripts/protocol/stage10_compare.py results/stage10/kaggle_bench --name stage10_benchmark
py -3 scripts/protocol/cost_projection.py --timing results/stage10/kaggle_bench/outcomes.csv \
    --label kaggle_gpu --runner-timing results/stage10/kaggle_bench/timing.json --weekly-quota-h <your quota>
```

`stage10_compare.py` applies the frozen agreement criteria (≥ 97% identical outcomes, accepted
positions within 5 m for ≥ 95%) and lists every disagreement. The cost sheet then decides whether
to stay on Kaggle (projection ≤ 3 × weekly quota).
