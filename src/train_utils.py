"""Helpers shared by train_clip_probe.py and train_resnet.py (Member B).

Two things every training run writes to Drive, in ONE fixed format so Member C's
evaluation code can read any run the same way:

1. A predictions CSV per run   results/predictions/<model>_<version>_seed<seed>.csv
   one row per val/test image:
   image_id, split, label, generator, source, score, model, data_version, seed
   - score = predicted probability that the image is FAKE (0..1)
   - label = true label (0 real, 1 fake), copied from splits.csv

2. One row per run in          results/experiment_log.csv
   (config, seed, data version, best epoch / best C, validation metrics, time)
   Validation metrics only: the test set is never used for any decision here.
"""
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def results_dir(cfg, override=None):
    """Folder on Drive where all of Member B's outputs go (created if missing)."""
    d = Path(override) if override else Path(cfg["drive_root"]) / cfg["train"]["results_dir"]
    (d / "predictions").mkdir(parents=True, exist_ok=True)
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    return d


def val_metrics(labels, scores):
    """Accuracy (threshold 0.5), AUC and log loss on the validation set."""
    labels, scores = np.asarray(labels), np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "val_acc": accuracy_score(labels, scores >= 0.5),
        "val_auc": roc_auc_score(labels, scores),
        "val_loss": log_loss(labels, scores, labels=[0, 1]),
    }


def save_predictions(rows_df, scores, out_path, model, version, seed):
    """rows_df = the splits.csv rows (val + test) in the same order as `scores`."""
    out = rows_df[["image_id", "split", "label", "generator", "source"]].copy()
    out["score"] = np.asarray(scores, dtype=float)
    out["model"], out["data_version"], out["seed"] = model, version, seed
    out.to_csv(out_path, index=False)
    return out


def log_run(log_path, row):
    """Append one run's summary to the experiment log (creates the header first time)."""
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    new = not Path(log_path).exists()
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def check_prediction_file(path, n_val=500, n_test=3000):
    """Quick sanity check of a predictions file (used by the --check option)."""
    p = pd.read_csv(path)
    assert (p.split == "val").sum() == n_val, "wrong number of val rows"
    assert (p.split == "test").sum() == n_test, "wrong number of test rows"
    assert p.score.between(0, 1).all(), "scores must be probabilities in [0, 1]"
    assert p.image_id.is_unique, "duplicate image ids"
    print(f"OK {Path(path).name}: {len(p)} rows, scores in [0,1], val={n_val}, test={n_test}")
