"""CLIP linear probe (Member B): a logistic-regression classifier on frozen CLIP features.

What it does, step by step:
  1. Load the CLIP ViT-B/32 features Member A saved (one 512-number vector per image).
  2. L2-normalise every vector (scale it to length 1), as in Ojha et al. (2023).
  3. Train logistic regression on the SD 1.4 TRAIN rows only.
  4. Try several regularisation strengths C; keep the one with the best VALIDATION accuracy.
  5. Score every val + test image with the chosen model and save the scores for Member C.
The test set is only scored, never used to choose anything.

Logistic regression has one best solution, so it gives the same answer for every
seed. We therefore run it once per data version (seed is recorded as "n/a").

Usage (from /content/repo, after the setup cells):
  python src/train_clip_probe.py --version original
  python src/train_clip_probe.py --version cleaned
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import load_config
from train_utils import results_dir, val_metrics, save_predictions, log_run

ap = argparse.ArgumentParser()
ap.add_argument("--version", required=True, choices=["original", "cleaned"])
ap.add_argument("--csv", default="splits/splits.csv")
ap.add_argument("--features", default=None, help="override path to the .npz (for testing)")
ap.add_argument("--out", default=None, help="override results folder (for testing)")
args = ap.parse_args()

cfg = load_config()
pcfg = cfg["train"]["clip_probe"]
out_dir = results_dir(cfg, args.out)
t0 = time.time()

# ---- 1. load features and check they line up with splits.csv ----------------------------
npz = args.features or Path(cfg["drive_root"]) / "features" / "clip_vitb32" / f"{args.version}.npz"
z = np.load(npz, allow_pickle=True)
df = pd.read_csv(args.csv)
assert (z["image_ids"] == df.image_id.values).all(), "feature rows are not in splits.csv order"
assert (z["labels"] == df.label.values).all(), "feature labels differ from splits.csv"

# ---- 2. L2-normalise: divide each vector by its length ---------------------------------
X = z["feats"].astype(np.float64)
X = X / np.linalg.norm(X, axis=1, keepdims=True)
y = df.label.values

train = (df.split == "train").values
val = (df.split == "val").values
evalrows = (df.split != "train").values          # val + test rows get predictions

# ---- 3 + 4. fit one model per C, keep the best on validation ---------------------------
best = None
for C in pcfg["C_grid"]:
    clf = LogisticRegression(C=C, max_iter=pcfg["max_iter"])
    clf.fit(X[train], y[train])
    m = val_metrics(y[val], clf.predict_proba(X[val])[:, 1])   # column 1 = P(fake)
    print(f"C={C:<6} val_acc={m['val_acc']:.4f} val_auc={m['val_auc']:.4f} val_loss={m['val_loss']:.4f}")
    # best = highest val accuracy; ties broken by lower val loss
    if best is None or (m["val_acc"], -m["val_loss"]) > (best[1]["val_acc"], -best[1]["val_loss"]):
        best = (C, m, clf)

C, m, clf = best
print(f"selected C={C}")

# ---- 5. score val + test and save -------------------------------------------------------
scores = clf.predict_proba(X[evalrows])[:, 1]
pred_path = out_dir / "predictions" / f"clip_probe_{args.version}.csv"
save_predictions(df[evalrows], scores, pred_path, "clip_probe", args.version, "n/a")
log_run(out_dir / "experiment_log.csv", {
    "model": "clip_probe", "data_version": args.version, "seed": "n/a",
    "setting": f"C={C}", "best_epoch": "n/a",
    "val_acc": round(m["val_acc"], 4), "val_auc": round(m["val_auc"], 4),
    "val_loss": round(m["val_loss"], 4), "minutes": round((time.time() - t0) / 60, 2),
    "predictions": pred_path.name,
})
print("saved", pred_path)
