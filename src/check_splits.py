"""Prove no file appears in two splits. Exits with an error if any check fails.
Usage: python src/check_splits.py --csv splits/splits.csv --root /content/data"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="splits/splits.csv")
ap.add_argument("--root", default=None)   # if given, also check files exist
args = ap.parse_args()
df = pd.read_csv(args.csv)

assert df.image_id.is_unique, "duplicate image_id"
assert df.path_original.is_unique, "duplicate path"
assert df.md5.is_unique, "same file content appears twice"
splits = {s: set(g.md5) for s, g in df.groupby("split")}
names = sorted(splits)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        both = splits[a] & splits[b]
        assert not both, f"{len(both)} files appear in both {a} and {b}"
assert set(df.split) <= {"train", "val", "test"}
assert set(df[df.split.isin(["train", "val"])].generator) <= {"real", "sd14"}, \
    "train/val contains a non-SD1.4 generator"
assert not (set(df[df.split == "test"].generator) & {"sd14"}), "SD1.4 fakes leaked into test"
if args.root:
    missing = [p for p in df.path_original if not (Path(args.root) / p).exists()]
    assert not missing, f"{len(missing)} files missing, e.g. {missing[:3]}"
print("ALL CHECKS PASSED:", len(df), "images,", dict(df.split.value_counts()))