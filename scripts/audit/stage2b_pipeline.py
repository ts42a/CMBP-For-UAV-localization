"""Retry download until complete, then extract and run full_dataset_audit."""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"F:\CMBL")
SCRIPTS = ROOT / "scripts" / "audit"
ARCHIVE = ROOT / "datasets" / "raw" / "UAV_VisLoc_dataset.zip"
LOG = ROOT / "datasets" / "raw" / "stage2b_pipeline.log"
MAX_ATTEMPTS = 48
WAIT_SEC = 600


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_py(name: str) -> int:
    p = SCRIPTS / name
    log(f"RUN {p}")
    return subprocess.call([sys.executable, "-u", str(p)])


def main() -> int:
    if ARCHIVE.exists() and ARCHIVE.stat().st_size >= 17_000_000_000:
        log("ARCHIVE already present; skip download")
    else:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            log(f"DOWNLOAD attempt {attempt}/{MAX_ATTEMPTS}")
            rc = run_py("download_full_release.py")
            if rc == 0 and ARCHIVE.exists():
                break
            if rc == 2:
                log(f"Drive blocked; sleep {WAIT_SEC}s")
                time.sleep(WAIT_SEC)
                continue
            log(f"download failed rc={rc}; sleep {WAIT_SEC}s")
            time.sleep(WAIT_SEC)
        else:
            log("GIVE UP download")
            return 1

    if run_py("extract_full_release.py") != 0:
        return 2
    if run_py("full_dataset_audit.py") != 0:
        return 3
    log("PIPELINE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
