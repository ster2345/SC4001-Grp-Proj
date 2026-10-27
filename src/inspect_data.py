"""Per-group stats: count, formats, sizes, rough JPEG quality.
Usage: python src/inspect_data.py --root /content/data/original --out stats.csv
Expects root/<group>/... where group is real, sd14, adm, etc."""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import list_images


def info(p):
    with Image.open(p) as im:
        fmt, (w, h) = im.format, im.size
        q = None
        if fmt == "JPEG" and getattr(im, "quantization", None):
            t = im.quantization[0]
            q = sum(t) / len(t)  # mean quant value: LOWER = higher quality
    return fmt, w, h, q


ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", default="stats.csv")
args = ap.parse_args()

rows = []
for group in sorted(d for d in Path(args.root).iterdir() if d.is_dir()):
    files = list_images(group)
    recs = [info(p) for p in tqdm(files, desc=group.name)]
    fmts = Counter(r[0] for r in recs)
    sizes = Counter((r[1], r[2]) for r in recs)
    qs = [r[3] for r in recs if r[3] is not None]
    rows.append({
        "group": group.name,
        "n": len(recs),
        "formats": dict(fmts),
        "n_unique_sizes": len(sizes),
        "top_sizes": sizes.most_common(3),
        "mean_quant_value": round(sum(qs) / len(qs), 2) if qs else None,
    })

df = pd.DataFrame(rows)
print(df.to_string())
df.to_csv(args.out, index=False)