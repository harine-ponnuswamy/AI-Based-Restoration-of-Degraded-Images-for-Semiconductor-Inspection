"""
Evaluation metrics: SSIM, PSNR, LPIPS — the exact numbers required for
Slide 6 of the Idea Submission deck. Instrumented here so they're logged
every validation epoch during training, not computed once at the end.
"""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

_lpips_model = None


def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net="alex")
        _lpips_model.eval()
    return _lpips_model


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    pred = np.clip(pred, 0, data_range)
    target = np.clip(target, 0, data_range)
    return float(sk_ssim(target, pred, data_range=data_range))


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    pred = np.clip(pred, 0, data_range)
    target = np.clip(target, 0, data_range)
    return float(sk_psnr(target, pred, data_range=data_range))


def compute_lpips(pred: np.ndarray, target: np.ndarray) -> float:
    """pred, target: 2D numpy arrays in [0, 1]. Returns scalar LPIPS distance.
    Requires internet access on first call to download AlexNet weights."""
    model = _get_lpips()

    def to_tensor(x):
        t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
        t = t.repeat(1, 3, 1, 1)
        return t * 2 - 1  # LPIPS expects inputs in [-1, 1]

    with torch.no_grad():
        d = model(to_tensor(pred), to_tensor(target))
    return float(d.item())


def evaluate_batch(preds: np.ndarray, targets: np.ndarray, data_range: float = 1.0, use_lpips: bool = True) -> dict:
    """preds, targets: numpy arrays shaped (B, H, W), values in [0, data_range]."""
    ssims, psnrs, lpipss = [], [], []
    for p, t in zip(preds, targets):
        ssims.append(compute_ssim(p, t, data_range))
        psnrs.append(compute_psnr(p, t, data_range))
        if use_lpips:
            try:
                lpipss.append(compute_lpips(p / data_range, t / data_range))
            except Exception as e:  # noqa: BLE001 - soft dependency, see compute_lpips docstring
                print(f"[metrics] LPIPS unavailable ({e}), skipping for this batch")
                use_lpips = False
    result = {"ssim": float(np.mean(ssims)), "psnr": float(np.mean(psnrs))}
    if use_lpips and lpipss:
        result["lpips"] = float(np.mean(lpipss))
    return result


if __name__ == "__main__":
    np.random.seed(0)
    target = np.random.rand(64, 64).astype(np.float32)
    pred_good = np.clip(target + np.random.randn(64, 64).astype(np.float32) * 0.01, 0, 1)
    pred_bad = np.random.rand(64, 64).astype(np.float32)

    print("Near-identical prediction:")
    print(f"  SSIM {compute_ssim(pred_good, target):.4f}  PSNR {compute_psnr(pred_good, target):.2f}")
    print("Random (bad) prediction:")
    print(f"  SSIM {compute_ssim(pred_bad, target):.4f}  PSNR {compute_psnr(pred_bad, target):.2f}")
    assert compute_ssim(pred_good, target) > compute_ssim(pred_bad, target)
    assert compute_psnr(pred_good, target) > compute_psnr(pred_bad, target)
    print("SSIM/PSNR self-test passed (good prediction scores higher, as expected)")

    try:
        d_good = compute_lpips(pred_good, target)
        d_bad = compute_lpips(pred_bad, target)
        print(f"LPIPS good={d_good:.4f} bad={d_bad:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"LPIPS self-test skipped (no internet for weights in this environment): {e}")
