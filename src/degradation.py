"""
Synthetic degradation for training-time augmentation
(see PROJECT_PLAN.md section 5).

The provided paired dataset gives one fixed degradation strength per
image. Generalizing to an out-of-distribution test set requires the
model to have seen a *range* of degradation strengths and orderings,
not just the one baked into the provided pairs. These functions
re-synthesize a degraded input from a ground-truth image with
randomized noise levels, randomized downsampling method, and randomized
ordering — used by PairedRestorationDataset to occasionally replace the
given NoisyLR sample with a freshly synthesized one.
"""
import numpy as np
from skimage.transform import resize


def add_speckle_noise(img: np.ndarray, var_range=(0.01, 0.08)) -> np.ndarray:
    """Multiplicative speckle noise: img + img * noise. This is what
    pushes pixel values outside the original signal range."""
    var = np.random.uniform(*var_range)
    noise = np.random.randn(*img.shape).astype(np.float32) * np.sqrt(var)
    return img + img * noise


def add_gaussian_noise(img: np.ndarray, sigma_range=(0.01, 0.06)) -> np.ndarray:
    """Additive Gaussian noise: softens edges, loses fine structure."""
    sigma = np.random.uniform(*sigma_range)
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return img + noise


def downsample(img: np.ndarray, factor: int = 2, order: int = None) -> np.ndarray:
    """Downsample by `factor`. order: 0=nearest, 1=bilinear, 3=bicubic.
    Randomized when not specified, since real-world degradation may use
    any of these."""
    if order is None:
        order = int(np.random.choice([0, 1, 3]))
    h, w = img.shape
    out = resize(
        img,
        (h // factor, w // factor),
        order=order,
        anti_aliasing=(order != 0),
        preserve_range=True,
    )
    return out.astype(np.float32)


def synthesize_degraded(
    gt_img: np.ndarray,
    scale: int = 2,
    apply_speckle: bool = True,
    apply_gaussian: bool = True,
    noise_first: bool = None,
) -> np.ndarray:
    """
    Build a synthetic degraded (noisy, low-res) image from a clean
    ground-truth image, with randomized degradation order and strength.

    The brief notes real degradations "may appear in any order" — so
    noise-before-downsample and downsample-before-noise are both
    represented, chosen randomly per call.
    """
    img = gt_img.astype(np.float32).copy()

    if noise_first is None:
        noise_first = bool(np.random.rand() < 0.5)

    def _apply_noise(x):
        if apply_speckle and np.random.rand() < 0.7:
            x = add_speckle_noise(x)
        if apply_gaussian and np.random.rand() < 0.5:
            x = add_gaussian_noise(x)
        return x

    if noise_first:
        img = _apply_noise(img)
        img = downsample(img, factor=scale)
    else:
        img = downsample(img, factor=scale)
        img = _apply_noise(img)

    return img


if __name__ == "__main__":
    # self-test with a synthetic image
    np.random.seed(0)
    fake_gt = np.random.rand(256, 256).astype(np.float32)
    degraded = synthesize_degraded(fake_gt, scale=2)
    print(f"GT shape {fake_gt.shape} -> degraded shape {degraded.shape}")
    print(f"GT range [{fake_gt.min():.3f}, {fake_gt.max():.3f}]")
    print(f"degraded range [{degraded.min():.3f}, {degraded.max():.3f}]  (may exceed GT range — expected)")
    assert degraded.shape == (128, 128)
    print("self-test passed")
