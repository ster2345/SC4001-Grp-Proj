"""Recompress + resize every image in splits.csv with ONE identical pipeline.
Usage:
  python src/clean_images.py --root /content/data --test      # 2 images per group, ~10 total
  python src/clean_images.py --root /content/data             # all images
  python src/clean_images.py --root /content/data --verify    # check outputs
(add --test to --verify to check only the test subset)"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/content/data")
ap.add_argument("--csv", default="splits/splits.csv")
ap.add_argument("--test", action="store_true")
ap.add_argument("--verify", action="store_true")
args = ap.parse_args()

cfg = load_config()
SIZE, Q = cfg["cleaning"]["size"], cfg["cleaning"]["jpeg_quality"]
root = Path(args.root)
df = pd.read_csv(args.csv)
if args.test:
    df = df.groupby("generator").head(2)   # real, sd14, adm, biggan, midjourney x 2


def clean(src, dst):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    im = im.crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.BICUBIC)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=Q)


if not args.verify:
    for r in tqdm(df.itertuples(), total=len(df)):
        dst = root / r.path_cleaned
        if not dst.exists():                      # skip work already done
            clean(root / r.path_original, dst)
    print("cleaned", len(df), "images")
else:
    bad, rows, missing = [], [], 0
    for r in df.itertuples():
        p = root / r.path_cleaned
        if not p.exists():
            missing += 1
            continue
        with Image.open(p) as im:
            q = im.quantization[0]
            rows.append({"generator": r.generator, "format": im.format, "size": im.size,
                         "quant": round(sum(q) / len(q), 3)})
            if im.format != "JPEG" or im.size != (SIZE, SIZE):
                bad.append(str(p))
    out = pd.DataFrame(rows)
    print(out.groupby("generator").agg(n=("format", "size"), formats=("format", lambda x: set(x)),
                                      sizes=("size", lambda x: set(x)),
                                      quant=("quant", lambda x: set(x))).to_string())
    print("checked:", len(out), "| missing:", missing, "| wrong format/size:", len(bad))
    assert not bad, bad[:3]
    assert out.quant.nunique() == 1, "quantisation tables differ between files"
    print("VERIFY PASSED: all JPEG, all", SIZE, "x", SIZE, ", identical quality")