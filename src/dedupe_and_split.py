"""Hash files, drop duplicate images, build splits/splits.csv.
Usage: python src/dedupe_and_split.py --root /content/data
Expects <root>/original/{real,sd14,adm,biggan,midjourney}/<tag>__<name>"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

from common import list_images, load_config, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/content/data")
ap.add_argument("--out", default="splits/splits.csv")
args = ap.parse_args()

cfg = load_config()
set_seed(cfg["seed"])
S = cfg["samples"]
orig = Path(args.root) / "original"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []
for group in ["real", cfg["train_generator"]] + cfg["test_generators"]:
    for p in list_images(orig / group):
        src = p.name.split("__")[0]          # archive the file came from
        rows.append({
            "image_id": p.stem,
            "path_original": str(p.relative_to(args.root)),
            "path_cleaned": str(Path("cleaned") / group / (p.stem + ".jpg")),
            "label": 0 if group == "real" else 1,   # 0 = real, 1 = fake
            "generator": "real" if group == "real" else group,
            "source": src,                       # real images: which archive they came from
            "md5": md5(p),
        })
df = pd.DataFrame(rows)
print("files found:", len(df))

# Drop exact duplicates (keep the first). Reals can repeat across archives.
dup = df.duplicated("md5", keep="first")
print("exact duplicates dropped:", int(dup.sum()), "| by group:",
      df[dup].generator.value_counts().to_dict())
df = df[~dup].copy()

# Assign splits
df["split"] = ""
tr = cfg["train_generator"]
for label in (0, 1):
    pool = df[(df.source == tr) & (df.label == label)].sample(frac=1, random_state=cfg["seed"])
    n_tr, n_va = S["train_per_class"], S["val_per_class"]
    df.loc[pool.index[:n_tr], "split"] = "train"
    df.loc[pool.index[n_tr:n_tr + n_va], "split"] = "val"
for g in cfg["test_generators"]:
    df.loc[(df.source == g), "split"] = "test"      # the fakes AND the reals taken from that archive
df = df[df.split != ""]

df = df.sort_values(["split", "generator", "image_id"]).reset_index(drop=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, index=False)
print(df.groupby(["split", "generator"]).size().to_string())
print("wrote", args.out)