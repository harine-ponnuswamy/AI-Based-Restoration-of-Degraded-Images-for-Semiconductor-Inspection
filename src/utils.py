import glob
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_norm_stats(gt_dir: str, sample_n: int = 500) -> dict:
    """Fixed, dataset-wide normalization stats computed from GT only
    (see PROJECT_PLAN.md section 3 — per-image normalization would
    rescale the noise distortion itself). Uses 1st/99th percentile
    across a sample of files rather than raw min/max, to avoid a single
    outlier image skewing normalization for the whole dataset."""
    files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))[:sample_n]
    if not files:
        raise RuntimeError(f"No .npy files found in {gt_dir}")
    mins, maxs = [], []
    for f in files:
        a = np.load(f)
        mins.append(float(a.min()))
        maxs.append(float(a.max()))
    return {
        "gt_min": float(np.percentile(mins, 1)),
        "gt_max": float(np.percentile(maxs, 99)),
    }


def save_checkpoint(path, model, optimizer=None, epoch=0, norm_stats=None, extra=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "norm_stats": norm_stats,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


if __name__ == "__main__":
    set_seed(0)
    m = AverageMeter()
    m.update(1.0)
    m.update(3.0)
    assert m.avg == 2.0
    print("utils self-test passed")
