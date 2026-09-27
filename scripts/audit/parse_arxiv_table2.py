"""Parse UAV-VisLoc arXiv HTML TABLE II and geometry notes into a local evidence file."""
from __future__ import annotations

import re
from pathlib import Path

HTML = Path(r"F:\CMBL\sources\UAV-VisLoc_arxiv_html.html")
OUT = Path(r"F:\CMBL\sources\UAV-VisLoc_arxiv_table2_extract.txt")

html = HTML.read_text(encoding="utf-8", errors="replace")
m = re.search(r'id="S2\.T2".*?</table>', html, re.S)
lines = []
if m:
    t = m.group(0)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S)
    lines.append(f"TABLE II rows: {len(rows)}")
    for r in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
        clean = []
        for c in cells:
            c = re.sub(r"<sup.*?</sup>", "", c)
            c = re.sub(r"<[^>]+>", "", c)
            c = re.sub(r"\s+", " ", c).strip()
            clean.append(c)
        if clean:
            lines.append(" | ".join(clean))
else:
    lines.append("TABLE II not found")

# footnote on Phi
fm = re.search(r"Phi denotes.{0,400}", html, re.S)
if fm:
    s = re.sub(r"<[^>]+>", " ", fm.group(0))
    s = re.sub(r"\s+", " ", s).strip()
    lines.append("PHI_FOOTNOTE: " + s)

# height / altitude statements
for pat in [
    r"from 400 meters to 2,000 meters.{0,120}",
    r"shooting height.{0,120}",
    r"Google Maps.{0,200}",
    r"spatial resolution of 0\.3 m.{0,80}",
    r"0\.1 m to 0\.2 m.{0,80}",
]:
    mm = re.search(pat, html, re.I | re.S)
    if mm:
        s = re.sub(r"<[^>]+>", " ", mm.group(0))
        s = re.sub(r"\s+", " ", s).strip()
        lines.append("HIT: " + s)

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT)
print("\n".join(lines[:40]))
