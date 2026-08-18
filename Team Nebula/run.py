"""
KLA SemiCon AI Hackathon — restoration inference entry point.

Usage (required format):
    python run.py <input-dir> <output-dir>

Reads every .npy file in <input-dir>, restores it (denoising + 2x
super-resolution), and writes a restored .npy file with the same
filename to <output-dir> (created if it doesn't exist).

Runs fully offline: no internet access, no API keys, no downloads,
no user interaction, no manual configuration required. Model weights
are loaded from the local models/ folder next to this script.
"""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(SCRIPT_DIR, "models", "model_final.pt")


# ---------------------------------------------------------------------------
# Model definition (self-contained here so run.py has no dependency on the
# rest of this repo's package structure — it runs standalone).
# ---------------------------------------------------------------------------
class DenseBlock(nn.Module):
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], 1))
        return x + 0.2 * x4


class RRDB(nn.Module):
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.db1 = DenseBlock(channels, growth)
        self.db2 = DenseBlock(channels, growth)
        self.db3 = DenseBlock(channels, growth)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return x + 0.2 * out


class PixelShuffleUpsample(nn.Module):
    def __init__(self, channels, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale * scale, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class RestorationNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=64, growth=32,
                 num_blocks=8, scale=2):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(channels, growth) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.upsample = PixelShuffleUpsample(channels, scale=scale)
        self.tail = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.head(x)
        body_out = self.body_conv(self.body(feat))
        feat = feat + body_out
        up = self.upsample(feat)
        out = self.tail(up)
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return out + base


_DEFAULT_MODEL_KWARGS = dict(in_channels=1, out_channels=1, channels=64, growth=32, num_blocks=8, scale=2)


def load_model(weights_path, device):
    if not os.path.isfile(weights_path):
        print(f"[run] ERROR: weights file not found at {weights_path}", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(weights_path, map_location=device)
    model_cfg = ckpt.get("config", {})
    if isinstance(model_cfg, dict) and "model" in model_cfg:
        model_cfg = model_cfg["model"]
    if not model_cfg:
        model_cfg = _DEFAULT_MODEL_KWARGS
    model = RestorationNet(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm_stats = ckpt.get("norm_stats") or {"gt_min": 0.0, "gt_max": 1.0}
    return model, norm_stats


def restore_array(model, arr, lo, hi, device):
    """arr: 2D numpy array (H, W). Returns restored 2D numpy array, values in [0,1]."""
    norm = (arr.astype(np.float32) - lo) / max(hi - lo, 1e-8)
    t = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t).clamp(0.0, 1.0)
    return out.cpu().numpy()[0, 0].astype(np.float32)


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>", file=sys.stderr)
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"[run] ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] device: {device}")

    model, norm_stats = load_model(DEFAULT_WEIGHTS, device)
    lo, hi = norm_stats["gt_min"], norm_stats["gt_max"]
    print(f"[run] normalization stats: min={lo:.4f} max={hi:.4f}")

    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not files:
        print(f"[run] ERROR: no .npy files found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[run] found {len(files)} input images")

    times = []
    n_written = 0
    for f in files:
        arr = np.load(f)
        if arr.ndim == 3:
            arr = arr[..., 0] if arr.shape[-1] == 1 else arr.squeeze()

        t0 = time.time()
        restored = restore_array(model, arr, lo, hi, device)
        times.append(time.time() - t0)

        # safety: guarantee no NaN/Inf and values in [0, 1], as required
        restored = np.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0)
        restored = np.clip(restored, 0.0, 1.0).astype(np.float32)

        out_path = os.path.join(output_dir, os.path.basename(f))
        np.save(out_path, restored)
        n_written += 1

    avg_ms = float(np.mean(times)) * 1000 if times else 0.0
    print(f"[run] processed {n_written} images")
    print(f"[run] average inference time per image: {avg_ms:.2f} ms")
    print(f"[run] restored outputs written to {output_dir}")


if __name__ == "__main__":
    main()
