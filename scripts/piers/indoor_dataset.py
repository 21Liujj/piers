"""SPAN8 dataset for indoor LoRA (CPU-ready). Ignores 255, returns 8-class trainIds."""
from pathlib import Path
import json, random
import numpy as np
from PIL import Image

ROOT = Path("/root/autodl-tmp/data_extra/SPAN8_PREP")
SPAN8 = ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"]
IGNORE = 255


def load_manifest(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


class SPAN8Dataset:
    """Returns (rgb_float CHW in [0,1], label H/W uint8 with 255=ignore)."""

    def __init__(self, manifest_path, split="fit", augment=True, seed=1234):
        self.root = ROOT
        self.records = load_manifest(manifest_path)
        self.augment = augment
        self.rng = random.Random(seed)
        self.num_classes = 8
        self.classes = SPAN8

    def __len__(self):
        return len(self.records)

    def _load(self, rec):
        img = Image.open(self.root / rec["area"] / rec["image"]).convert("RGB")
        lab = Image.open(self.root / rec["area"] / rec["label"])
        return np.array(img), np.array(lab)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img, lab = self._load(rec)
        if self.augment:
            # horizontal flip is invalid for equirect (wrap-around) -> only use
            # photometric jitter. Kept deliberately conservative for panorama.
            if self.rng.random() < 0.5:
                img = (img.astype(np.float32) * self.rng.uniform(0.85, 1.15)).clip(0, 255)
            else:
                img = img.astype(np.float32)
            if self.rng.random() < 0.3:
                img = (img * self.rng.uniform(0.9, 1.1)).clip(0, 255)
        else:
            img = img.astype(np.float32)
        img = (img / 255.0).transpose(2, 0, 1)  # CHW
        return img, lab.astype(np.uint8), rec["id"]


def collate(batch):
    imgs = np.stack([b[0] for b in batch])
    labs = np.stack([b[1] for b in batch])
    ids = [b[2] for b in batch]
    return imgs, labs, ids


def class_hist(labels, num_classes=8):
    h = np.zeros(num_classes + 1, dtype=np.int64)
    for lab in labels:
        u, c = np.unique(lab, return_counts=True)
        for k, v in zip(u.tolist(), c.tolist()):
            if k < num_classes:
                h[k] += v
            else:
                h[num_classes] += v
    return h
