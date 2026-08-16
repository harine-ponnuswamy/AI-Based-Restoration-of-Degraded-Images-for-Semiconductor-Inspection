"""
Reproduces the full training run from scratch.

Usage:
    python train.py --config configs/config.yaml
"""
import argparse
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import PairedRestorationDataset
from src.losses import CombinedRestorationLoss
from src.metrics import evaluate_batch
from src.model import RestorationNet
from src.utils import AverageMeter, compute_norm_stats, load_checkpoint, save_checkpoint, set_seed


def build_split(noisy_dir: str, gt_dir: str, val_frac: float, seed: int):
    """Split matched filenames into train/val lists (not a shared Subset
    over one Dataset instance, so train and val can carry different
    augmentation settings independently)."""
    all_pairs = PairedRestorationDataset._discover_pairs(noisy_dir, gt_dir)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_pairs))
    val_len = max(1, int(len(all_pairs) * val_frac))
    val_idx = set(idx[:val_len].tolist())
    train_files = [f for i, f in enumerate(all_pairs) if i not in val_idx]
    val_files = [f for i, f in enumerate(all_pairs) if i in val_idx]
    return train_files, val_files


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)

    print("[train] computing normalization stats from GT...")
    norm_stats = compute_norm_stats(cfg["data"]["train_gt_dir"])
    print(f"[train] norm stats: {norm_stats}")

    train_files, val_files = build_split(
        cfg["data"]["train_noisy_dir"],
        cfg["data"]["train_gt_dir"],
        cfg["data"]["val_split"],
        cfg["train"]["seed"],
    )

    # Optional caps for constrained-compute environments (e.g. this sandbox).
    # Leave unset / null in configs/config.yaml for the real training run —
    # this only exists so a bounded demo run finishes in limited wall-clock time.
    max_train = cfg["data"].get("max_train_pairs")
    max_val = cfg["data"].get("max_val_pairs")
    if max_train:
        train_files = train_files[:max_train]
    if max_val:
        val_files = val_files[:max_val]

    print(f"[train] train pairs: {len(train_files)}, val pairs: {len(val_files)}")

    train_ds = PairedRestorationDataset(
        noisy_dir=cfg["data"]["train_noisy_dir"],
        gt_dir=cfg["data"]["train_gt_dir"],
        scale=cfg["data"]["scale"],
        patch_size=cfg["data"]["patch_size"],
        norm_stats=norm_stats,
        augment=True,
        resynth_prob=cfg["train"]["resynth_prob"],
        filenames=train_files,
    )
    val_ds = PairedRestorationDataset(
        noisy_dir=cfg["data"]["train_noisy_dir"],
        gt_dir=cfg["data"]["train_gt_dir"],
        scale=cfg["data"]["scale"],
        patch_size=None,  # evaluate on full images, not random crops
        norm_stats=norm_stats,
        augment=False,
        filenames=val_files,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=max(1, cfg["train"]["num_workers"] // 2),
        pin_memory=True,
    )

    model = RestorationNet(**cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model parameters: {n_params / 1e6:.2f}M")

    criterion = CombinedRestorationLoss(**cfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["num_epochs"])
    amp_enabled = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_ssim = -1.0
    resume_path = cfg["train"].get("resume_from")
    if resume_path and os.path.isfile(resume_path):
        ckpt = load_checkpoint(resume_path, model, optimizer, map_location=device)
        start_epoch = ckpt.get("epoch", -1) + 1
        best_ssim = ckpt.get("val_metrics", {}).get("ssim", -1.0)
        print(f"[train] resumed from {resume_path} at epoch {start_epoch}, best_ssim so far {best_ssim:.4f}")

    for epoch in range(start_epoch, cfg["train"]["num_epochs"]):
        model.train()
        loss_meter = AverageMeter()
        t0 = time.time()
        for i, (noisy, gt, _) in enumerate(train_loader):
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(noisy)
                pred = pred.clamp(0.0, 1.0)  # stabilizes SSIM loss — see README section 7 (NaN note)
                loss, _ = criterion(pred, gt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item(), noisy.size(0))
            if (i + 1) % cfg["train"]["log_every"] == 0:
                print(f"[epoch {epoch}] step {i + 1}/{len(train_loader)} loss {loss_meter.avg:.4f}")
        scheduler.step()

        if (epoch + 1) % cfg["train"]["val_every"] == 0:
            model.eval()
            all_preds, all_gts = [], []
            with torch.no_grad():
                for noisy, gt, _ in val_loader:
                    noisy = noisy.to(device)
                    pred = model(noisy).clamp(0, 1).cpu().numpy()[:, 0]
                    all_preds.append(pred)
                    all_gts.append(gt.numpy()[:, 0])
            all_preds = np.concatenate(all_preds, axis=0)
            all_gts = np.concatenate(all_gts, axis=0)
            metrics = evaluate_batch(all_preds, all_gts, data_range=1.0, use_lpips=True)
            elapsed = time.time() - t0
            lpips_str = f"LPIPS {metrics['lpips']:.4f}" if "lpips" in metrics else "LPIPS n/a"
            print(
                f"[epoch {epoch}] val SSIM {metrics['ssim']:.4f} PSNR {metrics['psnr']:.2f} "
                f"{lpips_str} | train loss {loss_meter.avg:.4f} | epoch time {elapsed:.1f}s"
            )

            if metrics["ssim"] > best_ssim:
                best_ssim = metrics["ssim"]
                save_checkpoint(
                    os.path.join(cfg["train"]["checkpoint_dir"], "model_final.pt"),
                    model,
                    optimizer,
                    epoch,
                    norm_stats,
                    extra={"config": cfg, "val_metrics": metrics},
                )
                ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "model_final.pt")
                print(f"[epoch {epoch}] new best SSIM {best_ssim:.4f} — checkpoint saved to {ckpt_path}")

        save_checkpoint(
            os.path.join(cfg["train"]["checkpoint_dir"], "model_last.pt"),
            model,
            optimizer,
            epoch,
            norm_stats,
            extra={"config": cfg},
        )

    print(f"[train] done. best val SSIM: {best_ssim:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
