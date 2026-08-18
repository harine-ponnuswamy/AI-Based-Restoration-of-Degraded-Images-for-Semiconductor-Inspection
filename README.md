# AI-Based Restoration of Degraded Images — KLA SemiCon AI Hackathon

A single-pass neural network that jointly removes speckle/Gaussian noise and reconstructs
resolution lost to downsampling from semiconductor inspection images, in one forward pass.

See `PROJECT_PLAN.md` for the full design rationale (architecture choices, loss design,
augmentation strategy, and why this approach was chosen over alternatives).

## Official submission entry point

For the exact `python run.py <input-dir> <output-dir>` format required by the
hackathon's final technical check, see the [`Team_Nebula/`](./Team_Nebula) folder —
it's a minimal, self-contained, offline-runnable package (weights included) separate
from the full training/research codebase documented below.

## 1. Setup

```bash
git clone <this-repo-url>
cd kla_restoration
pip install -r requirements.txt
```

Tested with Python 3.12. A CUDA GPU is strongly recommended for training (CPU works but is
slow); inference runs on CPU or GPU.

**Note on the perceptual loss and LPIPS metric:** both use pretrained ImageNet weights
(VGG16 and AlexNet respectively) downloaded automatically by `torchvision`/`lpips` on first
use. This requires normal internet access the first time you run `train.py` or
`evaluate.py` in a fresh environment — after that, weights are cached locally. If weights
cannot be downloaded, the perceptual loss term silently contributes 0 (logged once) and
LPIPS is skipped in metric reporting — training and inference still run correctly on the
remaining loss terms / metrics.


## 2. Data layout

Place data in this structure (or point `configs/config.yaml` at your own paths):

```
data/
├── train/
│   ├── NoisyLR/   000000.npy, 000001.npy, ...   (degraded, low-res, float32 .npy, HxW)
│   └── GT/        000000.npy, 000001.npy, ...   (clean, full-res, float32 .npy, matching filenames)
└── test/
    └── NoisyLR/   000000.npy, ...                (held-out test set, no GT)
```

Filenames in `NoisyLR/` and `GT/` must match exactly — the dataset loader pairs them by
filename and prints a warning listing any that don't have a match on either side.


## 3. Train

```bash
python train.py --config configs/config.yaml
```

This will:
1. Compute fixed, dataset-wide normalization stats from `GT/` (1st/99th percentile of
   min/max across the training set — not per-image, see `PROJECT_PLAN.md` section 3 for why).
2. Split matched pairs into train/val (`val_split` in the config).
3. Train `RestorationNet` (`src/model.py`) using the combined loss in `src/losses.py`,
   with degradation-randomization augmentation (`src/degradation.py`) applied at the
   probability set by `resynth_prob`.
4. Log SSIM / PSNR / LPIPS on the validation split every epoch.
5. Save the best-SSIM checkpoint to `weights/model_final.pt` and the latest to
   `weights/model_last.pt`.

Key hyperparameters live in `configs/config.yaml` — model size, loss term weights, batch
size, epochs, learning rate, augmentation probability, and patch size are all there rather
than hardcoded.


## 4. Evaluate / run inference

```bash
python evaluate.py --input_dir data/test/NoisyLR --output_dir outputs/restored_test
```

This is the standalone script used for benchmarking — it takes only an input directory and
output directory, loads the trained weights, runs inference on every `.npy` file found, and
writes the restored image (same filename) to the output directory. It also prints average
inference time per image.

Optional flags:
```bash
python evaluate.py --input_dir <dir> --output_dir <dir> \
    --weights weights/model_final.pt \
    --batch_size 8 \
    --device cuda   # or cpu; auto-detected if omitted
```

No manual edits are required to run this on a fresh clone — it resolves the default weights
path relative to the script location, not the working directory.


## 5. Repository structure

## 5. Repository structure

```
├── README.md                   # this file
├── PROJECT_PLAN.md              # full design rationale and decisions
├── requirements.txt
├── Team_Nebula/                  # official submission entry point (run.py format)
├── configs/
│   └── config.yaml               # all model/training/data hyperparameters
├── data/
│   ├── train/
│   │   ├── NoisyLR/                # degraded, low-res training images
│   │   └── GT/                      # clean, full-res ground truth
│   └── test/
│       └── NoisyLR/                 # held-out test set, no GT
├── src/
│   ├── model.py                   # RestorationNet: encoder -> RRDB backbone -> pixel-shuffle upsampler
│   ├── losses.py                    # Charbonnier + SSIM + edge + perceptual combined loss
│   ├── degradation.py                # synthetic noise/downsample functions for augmentation
│   ├── dataset.py                     # paired NoisyLR/GT loader, normalization, augmentation
│   ├── metrics.py                      # SSIM, PSNR, LPIPS
│   └── utils.py                         # seeding, checkpointing, normalization stat computation
├── train.py                    # reproduces training from scratch
├── evaluate.py                 # STANDALONE inference script — see section 4
├── weights/
│   └── model_final.pt           # trained weights (see note below on large files)
├── outputs/
│   └── restored_test/            # model output on the test set
└── notebooks/                  # exploratory work only, never load-bearing
```

## 6. Model architecture summary

Input (degraded, low-res) → shared encoder → RRDB residual backbone (denoising + detail
reconstruction happen jointly here) → pixel-shuffle upsampler → output, with a global
image-level residual connection around a bicubic-upsampled version of the input. One
forward pass handles speckle noise, Gaussian noise, and super-resolution simultaneously —
see `PROJECT_PLAN.md` section 2 for why this beats a two-stage denoise-then-upscale
pipeline on both quality and inference speed.

~4.4M parameters at the default config (`channels=64, num_blocks=8`) — well under typical
SR-network sizes, chosen deliberately given the hackathon's explicit inference-speed
benchmark.


## 7. Actual results from this repository's trained checkpoint

The included `weights/model_final.pt` was trained end-to-end on the real KLA-provided
paired dataset (3,200 NoisyLR/GT training pairs, verified 1:1 matched by filename,
128×128 → 256×256, scale=2), on a Google Colab T4 GPU using the full-size architecture
from `configs/config.yaml`:

- Architecture: `channels=64, growth=32, num_blocks=8` (~4.39M params)
- Trained on the full 3,200-pair dataset (2,880 train / 320 val split), 100 epochs
- `amp: false` (disabled after an early NaN-loss issue traced to numerical instability
  in the SSIM loss term under mixed precision — see the note below) and `lr: 5.0e-5`
  (lowered from the config default `2.0e-4` for stability)
- Perceptual loss and LPIPS enabled (both downloaded pretrained ImageNet weights
  successfully during the actual training run, unlike the constrained dev sandbox)

**Validation results (320 held-out pairs, full 128×128→256×256 images, no patching):**

| | SSIM | PSNR | LPIPS |
|---|---|---|---|
| Bicubic upsampling only (no model) | 0.524 | 22.73 dB | — |
| This trained model | **0.732** | **27.64 dB** | **0.297** |

Independently re-verified (not just taken from the training log): re-ran inference with
this exact checkpoint on a 40-pair sample of the same validation split and got SSIM 0.720,
PSNR 27.15 — consistent with the reported numbers, confirming the result is real and
reproducible rather than a logging artifact. See `outputs/before_after_gt_comparison.png`
for visual examples.

**Test set inference:** all 400 real test images in `data/test/NoisyLR` processed
successfully via `evaluate.py`, restored outputs written to `outputs/restored_test/`.
GPU inference time from the actual training run's `evaluate.py` call wasn't captured in
the shared log — re-run `evaluate.py` on the target hardware and note the printed
`average inference time per image` line for the exact figure before finalizing Slide 7.

### A note on the NaN issue encountered during training

Partway through training, loss values began printing as `nan`. Root cause: the custom
`SSIMLoss` in `src/losses.py` involves divisions that can become unstable under mixed
precision (`amp: true`). Fix applied: `amp` disabled and learning rate reduced. If
re-enabling mixed precision for speed, either clamp the SSIM loss's denominator away from
zero, or compute the SSIM term in full precision even inside an autocast block.
Additionally, clamping the model's raw output to `[0, 1]` before computing the loss (not
just at inference time) and adding gradient clipping (`torch.nn.utils.clip_grad_norm_`)
were both applied as extra stability measures and are recommended to keep in any future
training run.

### Optional: pushing further

100 epochs on the full dataset already gives a strong, submission-ready result. If more
time is available before the deadline:

1. Re-enable `amp: true` with the SSIM-loss fix above, for faster epochs
2. Increase `num_epochs` further and watch for a validation SSIM plateau
3. Try restoring the default `lr: 2.0e-4` once the SSIM instability fix is in place, for
   faster convergence

No code changes are needed beyond what's already in this repo — only `configs/config.yaml`
values, and the stability fix noted above if re-enabling `amp`.


## 8. Known limitations / honesty notes

- Training briefly hit a `nan` loss mid-run on GPU with mixed precision enabled — see
  section 7 above for the root cause and the fix already applied in this checkpoint's
  training run. Worth double-checking if you retrain with `amp: true`.
- The train/val split is random over the paired set, not stratified by data source, since
  no source-of-origin metadata was available. If source labels become available, switch to
  a stratified split (see `PROJECT_PLAN.md` section 6) for a validation score that better
  predicts OOD test performance.
- GPU inference time (for Slide 7) should be re-confirmed by re-running `evaluate.py` and
  reading its printed timing line — the number in the deck is a placeholder based on this
  dev sandbox's CPU timing, not the actual GPU figure.

## 9. Large file handling

If `weights/model_final.pt` exceeds GitHub's file size limits, use Git LFS:
```bash
git lfs install
git lfs track "weights/*.pt"
git add .gitattributes weights/model_final.pt
```
or host it externally (Google Drive / HuggingFace) and link it here, with a short note in
this README on where to download it and where to place it before running `evaluate.py`.

