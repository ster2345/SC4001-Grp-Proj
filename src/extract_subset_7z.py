"""Sample images from a split GenImage zip using 7z and a saved 7z listing.
Fakes (ai/) -> <out>/<tag>/, reals (nature/) -> <out>/real/, renamed <tag>__<name>.
Usage:
  python src/extract_subset_7z.py --archive "<final .zip>" --listing /content/adm_list.txt --tag adm --part val --n 10
"""
import argparse
import random
import re
import shutil
import subprocess
from pathlib import Path

from common import IMG_EXTS

ap = argparse.ArgumentParser()
ap.add_argument("--archive", required=True)   # the final .zip (7z finds the .z01 parts itself)
ap.add_argument("--listing", required=True)   # saved output of `7z l`
ap.add_argument("--tag", required=True)
ap.add_argument("--part", default="val")
ap.add_argument("--n", type=int, default=10)  # images per class
ap.add_argument("--out", default="/content/data/original")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed)

LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \.{4}A\s+\d+\s+\d+\s+(.+?)\s*$")
names = []
with open(args.listing, errors="ignore") as f:
    for line in f:
        m = LINE.match(line)
        if m and Path(m.group(1)).suffix.lower() in IMG_EXTS:
            names.append(m.group(1))

picks = {}
for kind in ("ai", "nature"):
    pool = sorted(n for n in names if f"/{args.part}/{kind}/" in n)
    print(f"{args.tag} {args.part}/{kind}: {len(pool)} images found")
    for n in random.sample(pool, min(args.n, len(pool))):
        picks[n] = kind

tmp = Path(f"/content/tmp_extract/{args.tag}")
tmp.mkdir(parents=True, exist_ok=True)
listfile = tmp / "picks.txt"
listfile.write_text("\n".join(picks) + "\n")

subprocess.run(["7z", "x", args.archive, f"-o{tmp}", "-y", "-spd", f"@{listfile}"], check=True)

for n, kind in picks.items():
    d = Path(args.out) / (args.tag if kind == "ai" else "real")
    d.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp / n), str(d / f"{args.tag}__{Path(n).name}"))
print("done")