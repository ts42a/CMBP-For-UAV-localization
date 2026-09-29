"""Download IntelliSensing UAV-VisLoc full release (Google Drive) to F: with resume."""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import gdown
from gdown.exceptions import FileURLRetrievalError

ROOT = Path(__file__).resolve().parents[2]  # repository root
RAW = ROOT / "datasets" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
DEST = RAW / "UAV_VisLoc_dataset.zip"
RECORD = RAW / "UAV-VisLoc_full_download_record.txt"
LOG = RAW / "UAV-VisLoc_full_download.log"
DRIVE_ID = "1xYODANyilEMM3CfWh85APwkTHQeLTcCT"
SOURCE_URL = f"https://drive.google.com/file/d/{DRIVE_ID}/view"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_record(**extra) -> None:
    base = {
        "source_url": SOURCE_URL,
        "gdown_uc": f"https://drive.google.com/uc?id={DRIVE_ID}",
        "drive_id": DRIVE_ID,
        "release_identity": "IntelliSensing/UAV-VisLoc GitHub full dataset (README ~16.4 GB; folders 01-11)",
        "resolved_filename": "UAV_VisLoc_dataset.zip",
        "local_archive": str(DEST),
        "publisher_checksum": "not reported",
        "expected_size_readme_gb": "16.4",
        "sha256_note": "local integrity only; publisher checksum not reported",
    }
    base.update(extra)
    RECORD.write_text("\n".join(f"{k}: {v}" for k, v in base.items()) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest().upper()


def main() -> int:
    t0 = time.time()
    try:
        import shutil

        free = shutil.disk_usage("F:\\").free
    except Exception:
        free = -1
    write_record(
        access_start=datetime.now(timezone.utc).isoformat(),
        capacity_F_free_bytes_before=free,
        capacity_F_free_GB_before=round(free / (1024**3), 3) if free >= 0 else "unknown",
        status="downloading",
        partial_bytes_before=(
            partials[0].stat().st_size
            if (partials := sorted(RAW.glob("UAV_VisLoc_dataset.zip*.part"))) and not DEST.exists()
            else (DEST.stat().st_size if DEST.exists() else 0)
        ),
    )
    # gdown may leave *.part until complete; resume from partial if present
    partials = sorted(RAW.glob("UAV_VisLoc_dataset.zip*.part"))
    if partials and not DEST.exists():
        log(f"PARTIAL present: {partials[0].name} bytes={partials[0].stat().st_size}")
    log(f"START dest={DEST} resume={DEST.exists() or bool(partials)}")
    try:
        out = gdown.download(
            id=DRIVE_ID,
            output=str(DEST),
            quiet=False,
            resume=True,
            retries=10,
        )
    except FileURLRetrievalError as e:
        write_record(
            status="blocked_drive_quota",
            access_end=datetime.now(timezone.utc).isoformat(),
            error=str(e).split("\n")[0][:200],
        )
        log(f"BLOCKED quota: {e}")
        return 2
    log(f"gdown returned: {out!r}")
    if not DEST.exists():
        write_record(status="failed_missing_file", access_end=datetime.now(timezone.utc).isoformat())
        log("FAIL missing dest")
        return 1
    size = DEST.stat().st_size
    log(f"SIZE_BYTES={size} SIZE_GB={size/(1024**3):.3f}")
    log("Computing local SHA256...")
    digest = sha256_file(DEST)
    (RAW / "UAV_VisLoc_dataset.zip.sha256.txt").write_text(digest + "\n", encoding="utf-8")
    write_record(
        access_end=datetime.now(timezone.utc).isoformat(),
        status="downloaded",
        size_bytes=size,
        size_gb=round(size / (1024**3), 3),
        sha256_local=digest,
        elapsed_sec=round(time.time() - t0, 1),
        capacity_F_free_bytes_after=shutil.disk_usage("F:\\").free,
    )
    log(f"SHA256={digest}")
    log("END ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
