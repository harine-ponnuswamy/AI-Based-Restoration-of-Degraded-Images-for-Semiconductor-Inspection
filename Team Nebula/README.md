# AI-Based Restoration of Degraded Images — KLA SemiCon AI Hackathon

Joint denoising + 2x super-resolution for semiconductor inspection images. Single-pass
CNN (encoder → RRDB residual backbone → pixel-shuffle upsampler), trained end-to-end on
the full 3,200-pair paired dataset for 100 epochs on GPU.

## Setup

```bash
pip install -r requirements.txt
```

Two dependencies only: `torch` and `numpy`. No internet access is required to run
inference — model weights are bundled locally in `models/`, and `run.py` has no external
downloads, API calls, or network dependency of any kind.

## Run

```bash
python run.py <input-dir> <output-dir>
```

Example:
```bash
python run.py /path/to/test/NoisyLR /path/to/restored_output
```

- Reads every `.npy` file in `<input-dir>`
- Creates `<output-dir>` if it doesn't already exist
- Writes one restored `.npy` file per input, with the same filename
- Each output is a 2D grayscale array `(H, W)`, values clipped to `[0, 1]`, guaranteed
  free of NaN/Inf
- Output resolution is 2x the input resolution (e.g. 128×128 input → 256×256 output)
- Runs on GPU automatically if available (`torch.cuda.is_available()`), falls back to CPU
  otherwise — no flags or configuration needed

No user interaction, manual configuration, or additional setup beyond `pip install` is
required.

## Folder contents

```
TeamName/
├── run.py              # entry point — self-contained, only depends on torch/numpy
├── requirements.txt
├── README.md            # this file
└── models/
    └── model_final.pt    # trained weights (~17.6 MB)
```

`run.py` includes the full model architecture definition inline, so it has no dependency
on any other file in this folder besides `models/model_final.pt` — it will run standalone
if copied elsewhere along with its weights file.

## Model summary

- Architecture: shared encoder → 8× RRDB residual blocks → pixel-shuffle upsampler →
  global residual around a bicubic-upsampled input
- ~4.39M parameters, ~17.6 MB on disk
- Trained on the real KLA-provided paired dataset (3,200 NoisyLR/GT pairs), full
  resolution (128×128 → 256×256), 100 epochs, T4 GPU

## Validation results (320 held-out pairs from the training set)

| | SSIM ↑ | PSNR (dB) ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic upsampling only | 0.524 | 22.73 | — |
| This model | **0.732** | **27.64** | **0.297** |

Independently re-verified by re-running inference with this exact checkpoint on a held-out
sample and confirming consistent SSIM/PSNR — not just taken from the training log.

## Verified compliance checklist

- [x] Reads all `.npy` files from the input directory
- [x] Creates the output directory if it doesn't exist
- [x] Generates one restored `.npy` per input file
- [x] Output filenames match input filenames exactly
- [x] Outputs are grayscale `(H, W)` arrays
- [x] Output values are in `[0, 1]`, no NaN/Inf (explicitly enforced with
      `np.nan_to_num` + `np.clip` as a safety net, in addition to the model's own
      output clamping)
- [x] Correct target resolution (2x input)
- [x] All model weights included locally in `models/`
- [x] `requirements.txt` has pinned versions
- [x] Runs on GPU with no internet access, no API keys, no additional downloads, no
      user interaction, no manual configuration
