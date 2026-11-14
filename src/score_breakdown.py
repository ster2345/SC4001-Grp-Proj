"""Score breakdown per test generator (Member B): explains AUC values below 0.5.

No training: reads the prediction files that already exist on Drive.
  - clean test images:    results/predictions/<model>_<version>[_seed<s>].csv   (train_*.py, test rows)
  - degraded test images: results/predictions/robustness/**/*.csv              (02_evaluation_robustness.ipynb)

For every model / training data / test condition / test generator it reports:
  real_called_fake_%  share of REAL images with score >= 0.5 (false alarms)
  fake_called_fake_%  share of FAKE images with score >= 0.5 (detection rate)
  mean_score_real / mean_score_fake   average "how fake" score of each group
  auc                 ranking quality; below 0.5 means real images score higher than fakes
ResNet rows are averaged over the 3 seeds. Output: results/metrics/score_breakdown.csv

Usage (from /content/repo, after mounting Drive; no GPU needed):
  python src/score_breakdown.py
"""
import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from common import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=None, help="override results folder (default: drive_root/results)")
args = ap.parse_args()
cfg = load_config()
RESULTS = Path(args.results) if args.results else Path(cfg["drive_root"]) / cfg["train"]["results_dir"]


def summarise(df, model, version, condition, seed):
    """One row per test generator for a single prediction file."""
    rows = []
    for gen, g in df.groupby("source"):            # source = test generator (adm / biggan / midjourney)
        real, fake = g[g.label == 0], g[g.label == 1]
        rows.append({
            "model": model, "version": version, "condition": condition, "seed": seed, "generator": gen,
            "real_called_fake_%": 100 * (real.score >= 0.5).mean(),
            "fake_called_fake_%": 100 * (fake.score >= 0.5).mean(),
            "mean_score_real": real.score.mean(),
            "mean_score_fake": fake.score.mean(),
            "auc": roc_auc_score(g.label, g.score),
        })
    return rows


rows = []
# 1) clean test images: main prediction files from the training scripts (test rows only)
for f in glob.glob(str(RESULTS / "predictions" / "*.csv")):
    name = os.path.basename(f)[:-4]
    m = re.match(r"(clip_probe|resnet50)_(original|cleaned)(?:_seed(\d+))?$", name)
    if not m:                                      # skips ablation (_ep15) and test-run (_quick) files
        continue
    p = pd.read_csv(f)
    rows += summarise(p[p.split == "test"], "clip" if m[1] == "clip_probe" else "resnet",
                      m[2], "clean", m[3] or "n/a")

# 2) degraded test images: robustness prediction files from 02_evaluation_robustness.ipynb
rob = glob.glob(str(RESULTS / "predictions" / "robustness" / "**" / "*.csv"), recursive=True)
print(f"found {len(rob)} robustness files (expect 40)")
assert rob, "no robustness files found: check that Drive is mounted and the results path"
for f in rob:
    name = os.path.basename(f)[:-4]
    m = re.match(r"(?:clip|resnet50)_(original|cleaned)(?:_seed(\d+))?_(jpeg\d+|resize\d+)$", name)
    if not m:
        print("skipped (unexpected name):", name)
        continue
    model = "clip" if name.startswith("clip") else "resnet"
    rows += summarise(pd.read_csv(f), model, m[1], m[3], m[2] or "n/a")

# average ResNet over its seeds so each setting appears once
cols = ["real_called_fake_%", "fake_called_fake_%", "mean_score_real", "mean_score_fake", "auc"]
t = (pd.DataFrame(rows).groupby(["model", "version", "condition", "generator"], as_index=False)[cols]
     .mean().round(3))
t["backwards"] = t.mean_score_real > t.mean_score_fake   # real images scored as more fake than fakes
order = ["clean", "jpeg95", "jpeg80", "jpeg60", "resize128", "resize64"]
t["condition"] = pd.Categorical(t.condition, order, ordered=True)
t = t.sort_values(["generator", "model", "version", "condition"])

out = RESULTS / "metrics" / "score_breakdown.csv"
out.parent.mkdir(parents=True, exist_ok=True)
t.to_csv(out, index=False)
print(f"{len(t)} rows saved to {out}\n")
print("--- settings where AUC < 0.5 (scores point the wrong way) ---")
print(t[t.auc < 0.5].to_string(index=False))
