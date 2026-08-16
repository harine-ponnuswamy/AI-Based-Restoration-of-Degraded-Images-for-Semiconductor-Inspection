"""
Paired NoisyLR/GT dataset loader (see PROJECT_PLAN.md sections 3 and 5).

Directory layout expected:
    root/NoisyLR/000000.npy
    root/GT/000000.npy
Filenames must match between the two folders.

Key design choices:
  - Normalization uses FIXED, dataset-wide statistics (computed once from
    GT via utils.compute_norm_stats), applied identically to both input
    and target. Per-image min-max normalization would rescale the very
    noise distortion we're trying to remove — see PROJECT_PLAN.md section 3.
  - Degradation-randomization augmentation: with probability
    `resynth_prob`, the provided NoisyLR sample is replaced by a freshly
    synthesized degraded image (random noise, random downsample method,
    random order) built from the paired GT. This is the main lever for
    out-of-distribution generalization.
  - Train and validation must NOT share a mutable `augment` flag on one
    underlying dataset instance — this class takes an explicit filename
    list so callers can construct two separate instances (train augmented,
    val clean) over a disjoint split.
"""
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.degradation import synthesize_degraded


class PairedRestorationDataset(Dataset):
    def __init__(
        self,
        noisy_dir: str,
        gt_dir: str,
        scale: int = 2,
        patch_size: int = None,
        norm_stats: dict = None,
        augment: bool = True,
        resynth_prob: float = 0.3,
        filenames: list = None,
    ):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.resynth_prob = resynth_prob

        if filenames is not None:
            self.filenames = filenames
        else:
            self.filenames = self._discover_pairs(noisy_dir, gt_dir)

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No matching NoisyLR/GT pairs found between {noisy_dir} and {gt_dir}. "
                f"Check that filenames match exactly (e.g. NoisyLR/000000.npy <-> GT/000000.npy)."
            )

        self.norm_stats = norm_stats or {"gt_min": 0.0, "gt_max": 1.0}

    @staticmethod
    def _discover_pairs(noisy_dir: str, gt_dir: str) -> list:
        noisy_files = {os.path.basename(f) for f in glob.glob(os.path.join(noisy_dir, "*.npy"))}
        gt_files = {os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*.npy"))}
        common = sorted(noisy_files & gt_files)
        missing_gt = sorted(noisy_files - gt_files)
        missing_noisy = sorted(gt_files - noisy_files)
        if missing_gt:
            print(f"[dataset] warning: {len(missing_gt)} NoisyLR files have no matching GT, skipped "
                  f"(e.g. {missing_gt[:3]})")
        if missing_noisy:
            print(f"[dataset] warning: {len(missing_noisy)} GT files have no matching NoisyLR, skipped "
                  f"(e.g. {missing_noisy[:3]})")
        return common

    def __len__(self):
        return len(self.filenames)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        lo, hi = self.norm_stats["gt_min"], self.norm_stats["gt_max"]
        return (arr - lo) / max(hi - lo, 1e-8)

    def _load(self, name: str):
        noisy = np.load(os.path.join(self.noisy_dir, name)).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, name)).astype(np.float32)
        return noisy, gt

    @staticmethod
    def _augment_geo(noisy: np.ndarray, gt: np.ndarray):
        if random.random() < 0.5:
            noisy, gt = np.fliplr(noisy).copy(), np.fliplr(gt).copy()
        if random.random() < 0.5:
            noisy, gt = np.flipud(noisy).copy(), np.flipud(gt).copy()
        k = random.choice([0, 1, 2, 3])
        if k:
            noisy, gt = np.rot90(noisy, k).copy(), np.rot90(gt, k).copy()
        return noisy, gt

    def _random_crop_pair(self, noisy: np.ndarray, gt: np.ndarray):
        ph = self.patch_size
        h, w = noisy.shape
        if h <= ph or w <= ph:
            return noisy, gt
        top = random.randint(0, h - ph)
        left = random.randint(0, w - ph)
        noisy_patch = noisy[top : top + ph, left : left + ph]
        gt_patch = gt[
            top * self.scale : (top + ph) * self.scale,
            left * self.scale : (left + ph) * self.scale,
        ]
        return noisy_patch, gt_patch

    def __getitem__(self, idx):
        name = self.filenames[idx]
        noisy, gt = self._load(name)

        if noisy.shape[0] * self.scale != gt.shape[0] or noisy.shape[1] * self.scale != gt.shape[1]:
            raise ValueError(
                f"{name}: NoisyLR shape {noisy.shape} does not match GT shape {gt.shape} "
                f"at scale={self.scale}. Check the scale config matches this pair's actual resolutions."
            )

        if self.augment and random.random() < self.resynth_prob:
            noisy = synthesize_degraded(gt, scale=self.scale)

        if self.augment:
            noisy, gt = self._augment_geo(noisy, gt)

        if self.patch_size is not None:
            noisy, gt = self._random_crop_pair(noisy, gt)

        noisy = self._normalize(noisy)
        gt = self._normalize(gt)

        noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)
        return noisy_t, gt_t, name


if __name__ == "__main__":
    # self-test using the REAL NoisyLR data — GT is a placeholder here
    # (synthetic) purely to exercise the code path end-to-end; this is
    # NOT a substitute for testing against real GT once available.
    import sys

    print("[dataset self-test] NOTE: using placeholder synthetic GT — real GT not yet available.")
    noisy_dir = "data/train/NoisyLR"
    files = sorted(glob.glob(os.path.join(noisy_dir, "*.npy")))[:8]
    if not files:
        print("No NoisyLR files found for self-test.")
        sys.exit(1)

    tmp_gt_dir = "data/_selftest_gt"
    os.makedirs(tmp_gt_dir, exist_ok=True)
    for f in files:
        noisy = np.load(f)
        # placeholder: upsample the noisy image itself to stand in for GT
        # shape, purely so the pipeline can be exercised end to end
        from skimage.transform import resize

        fake_gt = resize(noisy, (noisy.shape[0] * 2, noisy.shape[1] * 2), order=3, preserve_range=True)
        np.save(os.path.join(tmp_gt_dir, os.path.basename(f)), fake_gt.astype(np.float32))

    ds = PairedRestorationDataset(
        noisy_dir=noisy_dir,
        gt_dir=tmp_gt_dir,
        scale=2,
        patch_size=64,
        norm_stats={"gt_min": 0.0, "gt_max": 1.5},
        augment=True,
        resynth_prob=0.5,
        filenames=[os.path.basename(f) for f in files],
    )
    print(f"dataset length: {len(ds)}")
    noisy_t, gt_t, name = ds[0]
    print(f"sample '{name}': noisy {tuple(noisy_t.shape)}, gt {tuple(gt_t.shape)}")
    assert noisy_t.shape == (1, 64, 64)
    assert gt_t.shape == (1, 128, 128)

    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch_noisy, batch_gt, names = next(iter(loader))
    print(f"batch noisy {tuple(batch_noisy.shape)}, batch gt {tuple(batch_gt.shape)}")
    print("self-test passed (placeholder GT — rerun once real GT is in data/train/GT)")

    import shutil

    shutil.rmtree(tmp_gt_dir)
