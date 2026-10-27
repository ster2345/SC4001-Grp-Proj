"""Frozen CLIP image features for every row of splits.csv, in row order.
Saves shards to Drive (safe against disconnects), then merges into one .npz.
Usage:
  python src/extract_clip.py --version original --test     # 10 images, quick check
  python src/extract_clip.py --version original            # all images
  python src/extract_clip.py --version cleaned
  python src/extract_clip.py --version original --verify   # check the saved .npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from common import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--version", required=True, choices=["original", "cleaned"])
ap.add_argument("--root", default="/content/data")
ap.add_argument("--csv", default="splits/splits.csv")
ap.add_argument("--shard", type=int, default=500)
ap.add_argument("--test", action="store_true")
ap.add_argument("--verify", action="store_true")
args = ap.parse_args()

cfg = load_config()
df = pd.read_csv(args.csv)
col = "path_original" if args.version == "original" else "path_cleaned"
feat_dir = Path(cfg["drive_root"]) / "features" / "clip_vitb32"
final = feat_dir / f"{args.version}.npz"

if args.verify:
    z = np.load(final, allow_pickle=True)
    print({k: z[k].shape for k in z.files})
    assert z["feats"].shape == (len(df), 512), z["feats"].shape
    assert np.isfinite(z["feats"]).all(), "NaN/inf in features"
    assert (z["image_ids"] == df.image_id.values).all(), "row order differs from splits.csv"
    assert (z["labels"] == df.label.values).all(), "labels differ from splits.csv"
    print("VERIFY PASSED:", final.name, z["feats"].shape)
    raise SystemExit

if args.test:
    df = df.groupby("generator").head(2).reset_index(drop=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| model:", cfg["clip"]["model"])
model = CLIPModel.from_pretrained(cfg["clip"]["model"]).to(dev).eval()
proc = CLIPProcessor.from_pretrained(cfg["clip"]["model"])


class DS(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(Path(args.root) / self.paths[i]).convert("RGB")
        return proc(images=im, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def encode(paths):
    out = []
    dl = DataLoader(DS(paths), batch_size=cfg["clip"]["batch_size"], num_workers=2)
    for x in dl:
        f = model.get_image_features(pixel_values=x.to(dev))
        if not torch.is_tensor(f):          # newer transformers versions return an object
            f = f.pooler_output
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


if args.test:
    f = encode(df[col].tolist())
    print("test feats:", f.shape, "| any NaN:", bool(np.isnan(f).any()),
          "| mean norm:", float(np.linalg.norm(f, axis=1).mean()))
    raise SystemExit

shard_dir = feat_dir / f"shards_{args.version}"
shard_dir.mkdir(parents=True, exist_ok=True)
parts = []
for i in tqdm(range(0, len(df), args.shard), desc=args.version):
    sp = shard_dir / f"shard_{i // args.shard:04d}.npy"
    if not sp.exists():                       # skip shards already done
        np.save(sp, encode(df[col].iloc[i:i + args.shard].tolist()))
    parts.append(np.load(sp))
feats = np.concatenate(parts)
assert len(feats) == len(df)

np.savez(final, feats=feats, labels=df.label.values,
         image_ids=df.image_id.values.astype(str), generators=df.generator.values.astype(str),
         sources=df.source.values.astype(str), splits=df.split.values.astype(str),
         paths=df[col].values.astype(str))
print("saved", final, feats.shape)