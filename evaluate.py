"""
Standalone evaluation script for the KLA SemiCon AI Hackathon restoration model.

This is the file KLA's benchmarking team runs AS-IS. It must work from a
fresh clone with no manual edits.

Usage:
    python evaluate.py --input_dir path/to/degraded_npy_dir --output_dir path/to/write_outputs
    python evaluate.py --input_dir data/test/NoisyLR --output_dir outputs/restored_test

Optional:
    --weights PATH      (default: weights/model_final.pt, relative to this script)
    --batch_size N       (default: 8)
    --device cuda|cpu    (default: auto-detected)

Loads the trained model, runs inference on every .npy image in input_dir,
and writes the restored image (same filename) to output_dir as .npy.
Also prints average inference time per image.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import RestorationNet

_DEFAULT_MODEL_KWARGS = dict(in_channels=1, out_channels=1, channels=64, growth=32, num_blocks=8, scale=2)


def load_model(weights_path: str, device: torch.device):
    ckpt = torch.load(weights_path, map_location=device)
    model_cfg = ckpt.get("config", {}).get("model", None) or _DEFAULT_MODEL_KWARGS
    model = RestorationNet(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm_stats = ckpt.get("norm_stats") or {"gt_min": 0.0, "gt_max": 1.0}
    return model, norm_stats


def _run_and_save(model, arr, lo, hi, filepath, output_dir, device, times):
    """Fallback path for a single image, used when a batch has mixed shapes."""
    norm = (arr - lo) / max(hi - lo, 1e-8)
    t = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model(t).clamp(0, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times.append(time.time() - t0)
    out_np = out.cpu().numpy()[0, 0]
    out_denorm = out_np * (hi - lo) + lo
    np.save(os.path.join(output_dir, os.path.basename(filepath)), out_denorm.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Run restoration inference on a directory of degraded images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of degraded .npy input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write restored .npy outputs")
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "model_final.pt"),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu, auto-detected if not set")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device: {device}")

    if not os.path.isfile(args.weights):
        print(f"[evaluate] ERROR: weights file not found at {args.weights}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    model, norm_stats = load_model(args.weights, device)
    lo, hi = norm_stats["gt_min"], norm_stats["gt_max"]
    print(f"[evaluate] normalization stats: min={lo:.4f} max={hi:.4f}")

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not files:
        print(f"[evaluate] ERROR: no .npy files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[evaluate] found {len(files)} input images")

    # warmup (first CUDA call includes init overhead, not representative of steady-state speed)
    first_shape = np.load(files[0]).shape
    dummy = torch.zeros(1, 1, *first_shape, device=device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    batch_size = args.batch_size
    for start in range(0, len(files), batch_size):
        batch_files = files[start : start + batch_size]
        arrs = [np.load(f).astype(np.float32) for f in batch_files]
        shapes = {a.shape for a in arrs}

        if len(shapes) > 1:
            # mixed shapes in this batch: fall back to one-by-one so a single
            # oddly-shaped file can't crash the whole run
            for f, a in zip(batch_files, arrs):
                _run_and_save(model, a, lo, hi, f, args.output_dir, device, times)
            continue

        norm = [(a - lo) / max(hi - lo, 1e-8) for a in arrs]
        batch = torch.from_numpy(np.stack(norm)).float().unsqueeze(1).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model(batch).clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        times.extend([elapsed / len(batch_files)] * len(batch_files))

        out_np = out.cpu().numpy()[:, 0]
        out_denorm = out_np * (hi - lo) + lo
        for f, o in zip(batch_files, out_denorm):
            out_path = os.path.join(args.output_dir, os.path.basename(f))
            np.save(out_path, o.astype(np.float32))

    avg_ms = float(np.mean(times)) * 1000
    print(f"[evaluate] processed {len(files)} images")
    print(f"[evaluate] average inference time per image: {avg_ms:.2f} ms")
    print(f"[evaluate] restored outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
