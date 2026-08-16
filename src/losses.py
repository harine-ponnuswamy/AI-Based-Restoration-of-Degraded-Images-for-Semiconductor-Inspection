"""
Combined restoration loss (see PROJECT_PLAN.md section 4).

Pure L1/L2 alone blurs fine detail — bad given the brief's explicit
"do not blur to remove noise" and "reconstruct fine detail" requirements.
This module combines four terms, each targeting a different failure mode:

  - Charbonnier: robust pixel fidelity (smooth L1, less outlier-sensitive
    than raw L1/L2)
  - SSIM: structural similarity — matches the Slide 6 grading metric directly
  - Edge (Sobel-gradient L1): forces sharp edges without the ringing
    artifacts a GAN loss risks
  - Perceptual (early VGG features): texture realism without hallucinating
    detail that isn't there

No adversarial loss — the classic source of ringing artifacts the brief
explicitly warns against, and not worth the training instability this
close to a deadline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):
    """Differentiable 1 - SSIM using a Gaussian window."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        window = self._gaussian_window(window_size, sigma)
        window = window.unsqueeze(0).unsqueeze(0)
        self.register_buffer(
            "window", window.expand(channels, 1, window_size, window_size).contiguous()
        )

    @staticmethod
    def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        return g.unsqueeze(1) @ g.unsqueeze(0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
        pad = self.window_size // 2
        window = self.window.to(dtype=pred.dtype)
        mu_p = F.conv2d(pred, window, padding=pad, groups=self.channels)
        mu_t = F.conv2d(target, window, padding=pad, groups=self.channels)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
        sigma_p2 = F.conv2d(pred * pred, window, padding=pad, groups=self.channels) - mu_p2
        sigma_t2 = F.conv2d(target * target, window, padding=pad, groups=self.channels) - mu_t2
        sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=self.channels) - mu_pt
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
            (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
        )
        return 1 - ssim_map.mean()


class EdgeLoss(nn.Module):
    """Sobel-gradient L1 loss — pushes edge sharpness without ringing."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _grad(self, x: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(dtype=x.dtype)
        ky = self.ky.to(dtype=x.dtype)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._grad(pred), self._grad(target))


class VGGPerceptualLoss(nn.Module):
    """Lightweight perceptual loss using early VGG16 features.
    Grayscale input is replicated to 3 channels for the pretrained net.
    Requires internet access on first use to download ImageNet weights —
    if unavailable, this term silently contributes zero (logged once)."""

    def __init__(self, layer_idx: int = 8):
        super().__init__()
        self.vgg = None
        try:
            from torchvision.models import VGG16_Weights, vgg16

            self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:layer_idx].eval()
            for p in self.vgg.parameters():
                p.requires_grad = False
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a soft dependency
            print(f"[losses] VGGPerceptualLoss: could not load pretrained VGG16 ({e}). "
                  f"Perceptual term will contribute 0 until weights are available.")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        p3 = pred.repeat(1, 3, 1, 1)
        t3 = target.repeat(1, 3, 1, 1)
        p3 = (p3 - self.mean) / self.std
        t3 = (t3 - self.mean) / self.std
        fp = self.vgg(p3)
        ft = self.vgg(t3)
        return F.l1_loss(fp, ft)


class CombinedRestorationLoss(nn.Module):
    def __init__(
        self,
        w_charbonnier: float = 1.0,
        w_ssim: float = 0.2,
        w_edge: float = 0.1,
        w_perceptual: float = 0.05,
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.perceptual = VGGPerceptualLoss() if use_perceptual else None
        self.w = dict(charbonnier=w_charbonnier, ssim=w_ssim, edge=w_edge, perceptual=w_perceptual)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        losses = {
            "charbonnier": self.charbonnier(pred, target),
            "ssim": self.ssim(pred, target),
            "edge": self.edge(pred, target),
        }
        losses["perceptual"] = (
            self.perceptual(pred, target) if self.perceptual is not None
            else torch.zeros((), device=pred.device, dtype=pred.dtype)
        )
        total = sum(self.w[k] * v for k, v in losses.items())
        losses["total"] = total
        return total, losses


if __name__ == "__main__":
    # self-test with dummy tensors
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)
    criterion = CombinedRestorationLoss(use_perceptual=True)
    total, parts = criterion(pred, target)
    print("loss breakdown:", {k: round(v.item(), 4) for k, v in parts.items()})
    total.backward()
    print("backward pass OK, grad exists:", pred.grad is not None)
