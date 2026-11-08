"""Fine-tuned ResNet-50 detector (Member B).

What it does, step by step:
  1. Load ResNet-50 pretrained on ImageNet and replace its last layer with ONE output
     (a "fake" score), so it becomes a real-vs-fake classifier.
  2. Train on the SD 1.4 TRAIN images (original or cleaned version), with Adam and
     binary cross-entropy loss.
  3. After every epoch, measure the loss on the VALIDATION images. Keep the weights from the
     epoch with the lowest validation loss; stop early if it hasn't improved for `patience`
     epochs (early stopping).
  4. Save the best weights to Drive, then score every val + test image and save the
     scores for Member C. The test set is only scored, never used to choose anything.

Image preprocessing (same for both data versions): resize the shorter side to 256, then take a
224x224 crop (random crop + horizontal flip when training; centre crop otherwise), then
normalise with ImageNet statistics. Resizing every image to the same scale stops the CNN
from simply reading image size, and matches CLIP, which also resizes to 224.

Usage (from /content/repo, on a T4 GPU runtime, after the setup cells):
  python src/train_resnet.py --version original --seed 42
  python src/train_resnet.py --version cleaned  --seed 42
  (repeat with --seed 43 and --seed 44)
  python src/train_resnet.py --version original --seed 42 --quick   # 64 images, 1 epoch: code check
"""
import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import load_config
from train_utils import results_dir, val_metrics, save_predictions, log_run

ap = argparse.ArgumentParser()
ap.add_argument("--version", required=True, choices=["original", "cleaned"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--root", default="/content/data")
ap.add_argument("--csv", default="splits/splits.csv")
ap.add_argument("--out", default=None, help="override results folder (for testing)")
ap.add_argument("--quick", action="store_true", help="tiny run to check the code works")
ap.add_argument("--no-pretrained", action="store_true", help="random init (offline testing only)")
args = ap.parse_args()

cfg = load_config()
rcfg = cfg["train"]["resnet"]
out_dir = results_dir(cfg, args.out)
dev = "cuda" if torch.cuda.is_available() else "cpu"
run_name = f"resnet50_{args.version}_seed{args.seed}" + ("_quick" if args.quick else "")


# ---- reproducibility: fix every random number generator ---------------------------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # same result every time (slightly slower)
    torch.backends.cudnn.benchmark = False


set_all_seeds(args.seed)

# ---- data --------------------------------------------------------------------------------
df = pd.read_csv(args.csv)
col = "path_original" if args.version == "original" else "path_cleaned"
if args.quick:   # a few images per split, just to check that everything runs end to end
    df = df.groupby(["split", "label"]).head(8).reset_index(drop=True)

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # ImageNet stats
train_tf = transforms.Compose([
    transforms.Resize(rcfg["resize"]), transforms.RandomCrop(rcfg["crop"]),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
eval_tf = transforms.Compose([
    transforms.Resize(rcfg["resize"]), transforms.CenterCrop(rcfg["crop"]),
    transforms.ToTensor(), norm])


class ImageDS(Dataset):
    """Reads the images listed in a slice of splits.csv; returns (image tensor, label)."""

    def __init__(self, rows, tf):
        self.paths = rows[col].tolist()
        self.labels = rows.label.values.astype(np.float32)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(Path(args.root) / self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


train_rows = df[df.split == "train"]
val_rows = df[df.split == "val"]
eval_rows = df[df.split != "train"]                      # val + test: get predictions

g = torch.Generator().manual_seed(args.seed)             # fixed shuffling order
loader_kw = dict(batch_size=rcfg["batch_size"], num_workers=rcfg["num_workers"])
train_dl = DataLoader(ImageDS(train_rows, train_tf), shuffle=True, generator=g, **loader_kw)
val_dl = DataLoader(ImageDS(val_rows, eval_tf), **loader_kw)
eval_dl = DataLoader(ImageDS(eval_rows, eval_tf), **loader_kw)

# ---- model: pretrained ResNet-50 with a 1-output head ------------------------------------
weights = None if args.no_pretrained else getattr(models.ResNet50_Weights, rcfg["weights"])
model = models.resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 1)            # 2048 features -> 1 "fake" logit
model = model.to(dev)

loss_fn = nn.BCEWithLogitsLoss()                         # sigmoid + binary cross-entropy
opt = torch.optim.Adam(model.parameters(), lr=rcfg["lr"])
use_amp = dev == "cuda"                                  # mixed precision: faster on T4
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


@torch.no_grad()
def predict(dl):
    """Return P(fake) for every image in the loader, in order."""
    model.eval()                                         # BatchNorm in eval mode
    out = []
    for x, _ in dl:
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out.append(torch.sigmoid(model(x.to(dev)).squeeze(1)).float().cpu())
    return torch.cat(out).numpy()


# ---- training loop with early stopping on validation loss --------------------------------
t0 = time.time()
max_epochs = 1 if args.quick else rcfg["max_epochs"]
best_loss, best_state, best_epoch, wait, best_m = float("inf"), None, 0, 0, None
for epoch in range(1, max_epochs + 1):
    model.train()
    running, n = 0.0, 0
    for x, y in train_dl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss = loss_fn(model(x).squeeze(1), y)       # forward pass + loss
        scaler.scale(loss).backward()                    # backpropagation
        scaler.step(opt)                                 # Adam update
        scaler.update()
        running += loss.item() * len(y)
        n += len(y)

    m = val_metrics(val_rows.label.values, predict(val_dl))
    print(f"epoch {epoch:2d}  train_loss {running / n:.4f}  val_loss {m['val_loss']:.4f}  "
          f"val_acc {m['val_acc']:.4f}  val_auc {m['val_auc']:.4f}", flush=True)

    if m["val_loss"] < best_loss - 1e-4:                 # improved: remember these weights
        best_loss, best_epoch, best_m, wait = m["val_loss"], epoch, m, 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        wait += 1
        if wait >= rcfg["patience"]:
            print(f"early stopping after epoch {epoch}")
            break

# ---- keep the best epoch, save checkpoint + predictions + log ----------------------------
model.load_state_dict(best_state)
ckpt = out_dir / "checkpoints" / f"{run_name}.pt"
torch.save({"state_dict": best_state, "epoch": best_epoch, "seed": args.seed,
            "version": args.version, "config": rcfg}, ckpt)

scores = predict(eval_dl)
pred_path = out_dir / "predictions" / f"{run_name}.csv"
save_predictions(eval_rows, scores, pred_path, "resnet50", args.version, args.seed)
log_run(out_dir / "experiment_log.csv", {
    "model": "resnet50" + ("_quick" if args.quick else ""), "data_version": args.version,
    "seed": args.seed, "setting": f"lr={rcfg['lr']},bs={rcfg['batch_size']}",
    "best_epoch": best_epoch, "val_acc": round(best_m["val_acc"], 4),
    "val_auc": round(best_m["val_auc"], 4), "val_loss": round(best_m["val_loss"], 4),
    "minutes": round((time.time() - t0) / 60, 2), "predictions": pred_path.name,
})
print(f"best epoch {best_epoch}; saved {ckpt.name} and {pred_path.name}")
