#!/usr/bin/env python3
"""Stage 10 benchmark runner for a Kaggle notebook (configs/stage10_benchmark.yaml).

Wires the read-only Kaggle inputs into the repo layout, fixes GPU numerics, runs the frozen
labeller on the 20 benchmark frames in-process, and records environment, timings and GPU
memory. Also times map feature extraction on region 10 and compares it with the PC cache.

In a Kaggle notebook cell (paths are examples; see kaggle/README.md):
  !python /kaggle/working/CMBL/kaggle/stage10_runner.py \\
      --code-root /kaggle/working/CMBL \\
      --data-root /kaggle/input/uav-visloc/UAV_VisLoc_dataset \\
      --emb-root /kaggle/input/cmbl-cache/embeddings \\
      --torch-home /kaggle/input/cmbl-cache/torch_home
Outputs go to <code-root>/results/stage10/kaggle_bench/ (outcomes.csv, run_manifest.json,
env.json, timing.json, runner.log). Copy that folder back to the PC repo.

Local mechanics test (CPU): add --device cpu --max-frames 2 --out results/stage10/local_mock.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import runpy
import shlex
import subprocess
import sys
import time
from pathlib import Path

BENCH_REGIONS = "01,02,03,04,05,06,08,09,10,11"


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)


def link(target: Path, link_path: Path) -> str:
    """Point link_path at target (symlink; Windows junction as a fallback)."""
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return "exists"
        raise SystemExit(f"{link_path} already exists and points elsewhere; remove it first")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link_path, target_is_directory=True)
        return "symlink"
    except OSError:
        if os.name != "nt":
            raise
        subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link_path), str(target)], stdout=subprocess.DEVNULL)
        return "junction"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--data-root", required=True, help="folder holding the region folders 01..11")
    ap.add_argument("--emb-root", required=True, help="folder holding region<RR>_dinov2_vits14_224.npz + .meta.json")
    ap.add_argument("--torch-home", required=True, help="TORCH_HOME with hub/facebookresearch_dinov2_main and hub/checkpoints")
    ap.add_argument("--out", default="results/stage10/kaggle_bench")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--max-frames", type=int, default=None, help="local mechanics test only")
    ap.add_argument("--skip-embed-bench", action="store_true")
    ap.add_argument("--label-args", default=None,
                    help="labeller arguments replacing the Stage 10 benchmark set, e.g. '--set full --regions 02' (Stage 11)")
    args = ap.parse_args()

    t_start = time.perf_counter()
    code = Path(args.code_root).resolve()
    out_dir = code / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "runner.log").open("a", encoding="utf-8")
    tee = Tee(sys.stdout, log)

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CMBL_ROOT"] = str(code)
    os.environ["TORCH_HOME"] = str(Path(args.torch_home).resolve())
    wiring = {
        "dataset": link(Path(args.data_root), code / "datasets" / "raw" / "UAV_VisLoc_dataset"),
        "embeddings": link(Path(args.emb_root), code / "datasets" / "cache" / "embeddings"),
    }

    t0 = time.perf_counter()
    import torch  # noqa: E402

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    import_s = time.perf_counter() - t0
    cuda = torch.cuda.is_available()
    try:
        smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        smi = None
    bundle_commit = (code / "BUNDLE_COMMIT").read_text(encoding="utf-8").strip() if (code / "BUNDLE_COMMIT").is_file() else "unknown"
    env = {
        "platform": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda_available": cuda, "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None, "nvidia_smi": smi,
        "cpu_count": os.cpu_count(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32, "bundle_commit": bundle_commit, "wiring": wiring,
        "device_arg": args.device, "label_args": args.label_args,
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps(env, indent=2), file=tee)

    sys.path.insert(0, str(code / "scripts" / "localization"))
    from dino_embedder import DinoGlobalEmbedder  # noqa: E402

    t0 = time.perf_counter()
    DinoGlobalEmbedder(image_size=224)
    model_load_s = time.perf_counter() - t0

    # Frozen labeller, in-process so the numerics flags above apply.
    frame_set = shlex.split(args.label_args) if args.label_args else [
        "--set", "pilot", "--per-region", "2", "--regions", BENCH_REGIONS]
    argv = ["label_windowed.py", *frame_set, "--out", args.out, "--resume"]
    if args.max_frames:
        argv += ["--max-frames", str(args.max_frames)]
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    old_argv, sys.argv = sys.argv, argv
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(tee):
            runpy.run_path(str(code / "scripts" / "label_windowed.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        sys.argv = old_argv
    label_s = time.perf_counter() - t0
    peak_label = torch.cuda.max_memory_allocated() / 2**20 if cuda else None

    embed = None
    if not args.skip_embed_bench:
        import numpy as np
        from dino_embedder import load_embedding_cache
        from project_paths import CACHE_EMB
        from region_map import load_region, read_tile_rgb_region
        from tile_index import TileSpec

        rasters, tiles = load_region("10", TileSpec(1024, 512))
        emb = DinoGlobalEmbedder(image_size=224)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        vecs, batch = [], []
        for i, t in enumerate(tiles):
            batch.append(read_tile_rgb_region(rasters, t))
            if len(batch) == 8 or i + 1 == len(tiles):
                vecs.append(emb.embed_rgb_arrays(batch, batch_size=8))
                batch = []
        if cuda:
            torch.cuda.synchronize()
        embed_s = time.perf_counter() - t0
        new = np.vstack(vecs)
        cached, _, _ = load_embedding_cache(CACHE_EMB / "region10_dinov2_vits14_224.npz")
        r = rasters[0]
        m_per_deg_lat, m_per_deg_lon = 110_574.0, 111_320.0 * np.cos(np.radians(r.north))
        km2 = sum(ra.height * ra.sy * m_per_deg_lat * ra.width * ra.sx * m_per_deg_lon for ra in rasters) / 1e6
        embed = {"region": "10", "tiles": len(tiles), "map_km2": round(km2, 3), "seconds": round(embed_s, 3),
                 "s_per_tile": round(embed_s / len(tiles), 4), "s_per_km2": round(embed_s / km2, 3),
                 "max_abs_diff_vs_pc_cache": float(np.abs(new - cached).max()),
                 "min_cosine_vs_pc_cache": float((new * cached).sum(1).min())}

    timing = {
        "import_torch_s": round(import_s, 3), "model_load_s": round(model_load_s, 3),
        "label_run_s": round(label_s, 3), "gpu_peak_mem_mib_labelling": peak_label,
        "embedding_bench": embed, "total_runner_s": round(time.perf_counter() - t_start, 3),
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(timing, indent=2), file=tee)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
