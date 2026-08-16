# KLA SemiCon AI Hackathon — Image Restoration: Full Project Plan

## 1. What we're actually solving

One model. Input: a grayscale image that is simultaneously noisy (speckle and/or Gaussian) and downsampled (2x, e.g. 128→256 or 256→512). Output: an image matching the clean, full-resolution ground truth. Must generalize to unseen semiconductor structure types (OOD test set) and must be fast at inference (benchmarked on H100).

Three things make this harder than a standard SR or denoising task:
- **Combined degradation, unknown mix.** A test image might have heavy noise + mild downsampling, or the reverse. The model can't assume a fixed degradation strength.
- **Speckle pushes pixel values out of range.** The input's intensity range can exceed the ground truth's. Naive min-max normalization per image will bake in that distortion — normalization needs to be range-aware.
- **Speed is graded, not just quality.** A slow diffusion or iterative model that scores well on SSIM/PSNR will lose to a fast one-shot CNN that scores adequately. Optimize for inference latency from day one, not as a post-hoc compression step.

## 2. Model architecture

**Recommendation: a single-pass CNN — not two separate models chained (denoise-then-upscale), not a diffusion model.**

Why not chain two models (e.g., a denoiser followed by a separate SR network)?
- Errors compound: the denoiser's artifacts become the SR model's input, and SR can't tell noise from signal it inherited.
- Two forward passes double inference time — directly hurts your speed score.
- Two models to train, tune, and justify in the "innovation" slide — more surface area for something to go wrong before the deadline.

Why not diffusion / iterative refinement?
- Quality can be excellent, but multi-step sampling is 10-100x slower than a single CNN forward pass. Given the explicit speed benchmark, this is a real risk unless heavily distilled — not worth it for the hackathon timeline.

**Architecture shape** (matches the diagram above):
- **Shared encoder**: a few conv + downsample blocks extracting features directly from the degraded low-res input — no upsampling first. (Upsampling before feature extraction, like old SRCNN-style bicubic pre-upsampling, wastes compute on interpolated garbage.)
- **Restoration backbone**: a stack of residual blocks (RRDB-style or NAFBlock-style) that do the heavy lifting — this is where denoising and detail reconstruction both happen, jointly, since the network doesn't need to know which degradation it's undoing.
- **Upsampler**: pixel-shuffle (sub-pixel convolution) to go from low-res feature maps to full-res output in one learned step — cheaper and sharper than transposed convolution or interpolate-then-conv.
- **Global residual / skip connection**: predict the *correction* on top of an upsampled version of the input rather than the raw pixels from scratch. This stabilizes training and speeds convergence.

Concrete starting point: a compact NAFNet or Real-ESRGAN-lite (RRDB, ~6-10 blocks, 64 channels, no batch norm — batch norm hurts restoration tasks, use no normalization or a lightweight alternative). Target **under 15M parameters** so a 256x256 image restores in well under a second on an H100.

## 3. Handling the "input range exceeds ground truth range" problem

Don't min-max normalize each input image independently — that rescales the very distortion you're trying to remove and teaches the model a moving target. Instead:
- Normalize using a **fixed, dataset-wide statistic** (e.g., ground truth's global min/max or mean/std across the training set), applied identically to both input and target.
- Let the input legitimately exceed [0, 1] after this normalization — that's expected and informative (it signals "there's noise here").
- Clip only the **final model output** to the valid image range as a postprocessing step, never the input.

## 4. Loss function design

Pure L1 or MSE alone will blur fine detail (the averaging behavior that L2 in particular is notorious for) — bad for both the "don't blur to remove noise" and "reconstruct fine detail" requirements. Combine:

| Term | Purpose | Weight (starting point) |
|---|---|---|
| Charbonnier loss (smooth L1) | Pixel-level fidelity, more robust to outliers than raw L1/L2 | 1.0 |
| SSIM loss (1 − SSIM) | Structural similarity, matches your Slide 6 metric directly | 0.2 |
| Edge / gradient loss (Sobel difference between output and target) | Forces sharp edges without the ringing artifacts a GAN loss risks | 0.1 |
| Lightweight perceptual loss (VGG features, low layer only) | Texture realism without hallucinating detail that isn't there | 0.05 |

Skip an adversarial (GAN) loss unless time permits — it's the classic source of the "ringing artifacts" the brief explicitly warns against, and it adds training instability you don't want this close to a deadline.

## 5. Data augmentation & generalization strategy

The OOD test set is the real differentiator — this is where most teams will lose points. Strategy: **degradation randomization**, not just geometric augmentation.

- **Geometric**: random crop, horizontal/vertical flip, 90° rotations (8-fold symmetry) — cheap, always safe for grayscale inspection imagery.
- **Degradation-space augmentation** (the important one): even though you're given paired data, re-synthesize additional training pairs by taking ground-truth images and applying randomized degradation — random speckle variance, random Gaussian noise sigma, random downsampling factor (bicubic/bilinear/nearest mixed) — so the model sees a continuous range of degradation strengths, not just the fixed strength in the provided pairs. This is what makes a model robust to a test set "from different sources."
- **Mixed degradation order**: randomize whether noise is applied before or after downsampling in your synthetic pairs, since the brief notes real degradations "may appear in any order."
- **Patch-based training**: train on random crops (e.g., 128x128 or 64x64 patches) rather than full images — increases effective dataset size and lets you use a larger batch size, which matters for a compact model like this.

## 6. Training strategy

- Mixed precision (fp16/bf16) training — faster, and forces you to confirm the model is numerically stable at inference precision too.
- Cosine learning-rate schedule with warmup; AdamW optimizer.
- Curriculum: start training on mild degradations (low noise, small downsampling factor), progressively introduce harder synthetic degradations over epochs — helps convergence versus throwing max difficulty at an untrained network immediately.
- Hold out a validation split stratified by data source (not random), so your own validation score actually predicts OOD generalization instead of just in-distribution accuracy.
- Log SSIM, PSNR, and LPIPS on validation every epoch — these are your Slide 6 numbers, so instrument this from the start rather than computing them once at the end.

## 7. Inference speed optimization

- Keep the model under ~15M params; profile actual inference time on the target hardware class (H100) early, not at the end.
- Export to ONNX or TorchScript for the evaluation script — faster and more portable than raw eager-mode PyTorch.
- Batch inference in the evaluation script wherever the test harness allows it, rather than looping image-by-image with per-call overhead.
- Report inference time **per image**, measured after a few warmup iterations (first-call CUDA initialization overhead isn't representative).

## 8. GitHub repository structure

```
repo/
├── README.md                  # setup + how to run inference, from a clean clone
├── requirements.txt            # exact pip freeze from training env
├── configs/
│   └── config.yaml             # architecture + training hyperparameters
├── src/
│   ├── model.py                 # network definition (encoder/backbone/upsampler)
│   ├── dataset.py                # paired dataset loader + degradation augmentation
│   ├── losses.py                  # Charbonnier + SSIM + edge + perceptual combo
│   ├── metrics.py                  # SSIM, PSNR, LPIPS computation
│   └── utils.py
├── train.py                    # reproduces training from scratch
├── evaluate.py                 # STANDALONE script — the most important file.
│                                #   python evaluate.py --input_dir <path> --output_dir <path>
│                                #   loads trained weights, runs inference, writes restored images.
│                                #   No manual edits required. No notebook.
├── weights/
│   └── model_final.pt          # or a download link in README if too large for git
├── outputs/
│   └── restored_test/          # your model's actual output on the test set
└── notebooks/                  # optional exploratory work — never load-bearing
```

**On `evaluate.py` specifically** (per the brief, this is what gets benchmarked as-is): argument parsing with `argparse`, hardcode nothing about paths, load weights relative to the script location, handle both single images and a directory of images, print per-image and average inference time, and test it yourself on a fresh clone/fresh environment before submitting — exactly as the brief warns.

## 9. Idea Submission PPT — slide-by-slide content plan

Using the provided template (9 slides max, remove instruction slide, save as PDF, name `TeamName_KLA_PS01.pdf`):

1. **Team Details** — team name, members, roles (e.g., who owns model/training vs. data pipeline vs. writeup), college, contact.
2. **Problem Statement Addressed** — "AI-Based Restoration of Degraded Images." One paragraph in your own words: why sharp inspection images matter for chip yield (a single hidden pixel can hide a fatal defect).
3. **Idea Description** — single unified restoration network handling speckle noise, Gaussian noise, and super-resolution jointly in one forward pass. Explain why joint > chained models (Section 2 above).
4. **Proposed Solution** — the architecture diagram from this plan (encoder → backbone → upsampler → output, loss branch), training strategy, loss function breakdown, augmentation approach.
5. **Innovation & Uniqueness** — degradation-randomization augmentation for OOD generalization, range-aware normalization for the speckle overflow issue, compact single-pass design chosen explicitly for inference speed rather than chasing SOTA quality at any cost.
6. **Results** — SSIM / PSNR / LPIPS table, before → after → ground-truth image triptychs (pick 3-4 representative examples, including at least one OOD-style case if you can construct one).
7. **Technology & Feasibility** — PyTorch, GPU used for training, training time, parameter count, model size (MB), measured inference time per image.
8. **GitHub & Video Link** — repo URL (mandatory), demo video link (optional).
9. **References** — any papers/architectures you drew from (e.g., RRDB/ESRGAN, NAFNet), and the dataset.

## 10. Suggested build order

1. Data loading + range-aware normalization + metrics (SSIM/PSNR/LPIPS) scaffolding first — you can't evaluate anything without this.
2. Baseline model (even a small plain CNN) trained end-to-end, to get the full pipeline (train → evaluate.py → metrics) working before optimizing architecture.
3. Swap in the fuller architecture (residual backbone + pixel shuffle) once the pipeline is proven.
4. Add degradation-randomization augmentation — this is your biggest lever on the OOD score.
5. Tune the loss weighting once training is stable.
6. Lock `evaluate.py` early and test it on a clean clone repeatedly — this is graded as-is and is the single highest-risk file in the whole submission.
7. Fill in the deck last, using real numbers from your actual runs.

