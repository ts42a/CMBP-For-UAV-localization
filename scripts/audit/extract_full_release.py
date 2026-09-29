"""Extract UAV_VisLoc_dataset.zip into datasets/full without overwriting sample/raw archives."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
ARCHIVE = ROOT / "datasets" / "raw" / "UAV_VisLoc_dataset.zip"
DEST = ROOT / "datasets" / "full"
DEST.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not ARCHIVE.exists():
        # gdown may leave .part until complete
        parts = list((ROOT / "datasets" / "raw").glob("UAV_VisLoc_dataset.zip*.part"))
        print("MISSING_ARCHIVE", ARCHIVE, "parts", parts)
        return 2
    print("ARCHIVE", ARCHIVE, "size", ARCHIVE.stat().st_size)
    # integrity: test zip central directory
    with zipfile.ZipFile(ARCHIVE, "r") as zf:
        bad = zf.testzip()
        names = zf.namelist()
        print("ZIP_OK", bad is None, "n_entries", len(names), "testzip", bad)
        # preview top-level
        tops = sorted({n.split("/")[0] for n in names if n})
        print("TOP_LEVEL", tops[:30], "...")
        # write listing (paths only) for audit
        listing = ROOT / "datasets" / "raw" / "UAV_VisLoc_dataset_zip_listing.txt"
        with listing.open("w", encoding="utf-8") as f:
            for n in names:
                info = zf.getinfo(n)
                f.write(f"{info.file_size}\t{n}\n")
        print("LISTING", listing)
        # extract; ZipFile.extractall won't overwrite? it will overwrite files.
        # Ensure DEST empty-ish except maybe prior partial — refuse if non-empty conflicting
        existing = [p for p in DEST.iterdir()]
        if existing:
            print("DEST_NONEMPTY", existing[:10])
            # still extract with exist_ok behavior: skip files that already exist and match size
        for info in zf.infolist():
            target = DEST / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
    print("EXTRACT_DONE", DEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
