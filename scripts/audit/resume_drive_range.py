"""Attempt HTTP Range resume on a previously observed Drive redirect URL (may expire)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

PART = Path(r"F:\CMBL\datasets\raw\UAV_VisLoc_dataset.zipi4suhujo.part")
DEST = Path(r"F:\CMBL\datasets\raw\UAV_VisLoc_dataset.zip")
TOTAL = 17662427174
URLS = [
    "https://drive.google.com/uc?id=1xYODANyilEMM3CfWh85APwkTHQeLTcCT&confirm=t&uuid=37541ffe-e27a-467b-b0fa-d658ec5fec16",
    "https://drive.google.com/uc?export=download&id=1xYODANyilEMM3CfWh85APwkTHQeLTcCT&confirm=t",
]

start = PART.stat().st_size if PART.exists() else 0
print("start", start, "total", TOTAL)
for url in URLS:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print("URL", url[:80], "status", r.status, "len", r.headers.get("Content-Length"), "cr", r.headers.get("Content-Range"))
            chunk = r.read(4096)
            print("first_bytes", len(chunk), chunk[:20])
    except Exception as e:
        print("FAIL", url[:60], type(e).__name__, e)
