"""Randomly sample images from one GenImage archive (.zip or .tar/.tar.gz).
Fakes (ai/) go to <out>/<tag>/, reals (nature/) go to <out>/real/.
Files are renamed <tag>__<original name> so reals from different archives never collide.
Usage:
  python src/extract_subset.py --archive "<path>" --tag sd14 --part train --n 30
"""
import argparse
import random
import tarfile
import zipfile
from pathlib import Path

from tqdm import tqdm

from common import IMG_EXTS

ap = argparse.ArgumentParser()
ap.add_argument("--archive", required=True)
ap.add_argument("--tag", required=True)            # sd14 / adm / biggan / midjourney
ap.add_argument("--part", default="val")           # train or val
ap.add_argument("--n", type=int, default=30)       # images per class (ai and nature)
ap.add_argument("--out", default="/content/data/original")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed)


def is_img(n):
    return Path(n).suffix.lower() in IMG_EXTS


def match(n, kind):
    return f"/{args.part}/{kind}/" in "/" + n.replace("\\", "/")


def save(kind, name, data):
    d = Path(args.out) / (args.tag if kind == "ai" else "real")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.tag}__{Path(name).name}").write_bytes(data)


def choose(names):
    picks = {}
    for kind in ("ai", "nature"):
        pool = sorted(n for n in names if match(n, kind))
        print(f"{args.tag} {args.part}/{kind}: {len(pool)} images found")
        for n in random.sample(pool, min(args.n, len(pool))):
            picks[n] = kind
    return picks


if args.archive.endswith(".zip"):
    with zipfile.ZipFile(args.archive) as z:
        picks = choose([n for n in z.namelist() if is_img(n)])
        for n, kind in tqdm(picks.items()):
            save(kind, n, z.read(n))
else:  # tar / tar.gz: pass 1 lists names, pass 2 extracts the chosen files
    with tarfile.open(args.archive) as t:
        picks = choose([n for n in t.getnames() if is_img(n)])
    with tarfile.open(args.archive) as t:
        for m in tqdm(t):
            if m.name in picks:
                save(picks[m.name], m.name, t.extractfile(m).read())
print("done")