import random
from pathlib import Path

import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)