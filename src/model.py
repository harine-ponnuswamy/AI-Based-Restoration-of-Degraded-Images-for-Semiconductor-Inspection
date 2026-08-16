"""
Restoration model: a single-pass network that jointly removes noise
(speckle + Gaussian) and reconstructs resolution lost to downsampling,
in one forward pass.

Design rationale (see PROJECT_PLAN.md section 2):
- One model, not a denoiser chained into a separate SR model — avoids
  compounding artifacts and halves inference cost.
- No batch norm — batch norm statistics don't transfer well across the
  varied degradation strengths this task requires, and restoration
  literature (ESRGAN, RRDB) consistently drops it.
- Global image-level residual: the network predicts a correction on top
  of a bicubic-upsampled version of the input rather than the image from
  scratch. This stabilizes early training and speeds convergence.
- Pixel-shuffle upsampling: cheaper and produces fewer checkerboard
  artifacts than transposed convolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """Residual dense block: 4 conv layers with dense skip connections
    (ESRGAN-style), residual-scaled to keep training stable."""

    def __init__(self, channels: int = 64, growth: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], 1))
        return x + 0.2 * x4


class RRDB(nn.Module):
    """Residual-in-residual dense block: 3 dense blocks with an outer
    residual connection. This is the workhorse of the restoration
    backbone — denoising and detail reconstruction both happen here,
    jointly, since the network is never told which degradation it's
    undoing."""

    def __init__(self, channels: int = 64, growth: int = 32):
        super().__init__()
        self.db1 = DenseBlock(channels, growth)
        self.db2 = DenseBlock(channels, growth)
        self.db3 = DenseBlock(channels, growth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return x + 0.2 * out


class PixelShuffleUpsample(nn.Module):
    """Sub-pixel convolution upsampling, single learned step."""

    def __init__(self, channels: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale * scale, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class RestorationNet(nn.Module):
    """
    Joint denoise + super-resolution network.

    Input:  degraded, low-resolution grayscale image  (B, in_channels, H, W)
    Output: restored, full-resolution grayscale image (B, out_channels, H*scale, W*scale)

    Handles speckle noise, Gaussian noise, and downsampling simultaneously —
    the model architecture makes no assumption about which degradation(s)
    are present or their strength.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 64,
        growth: int = 32,
        num_blocks: int = 8,
        scale: int = 2,
    ):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(channels, growth) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.upsample = PixelShuffleUpsample(channels, scale=scale)
        self.tail = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        body_out = self.body_conv(self.body(feat))
        feat = feat + body_out  # feature-level residual around the backbone
        up = self.upsample(feat)
        out = self.tail(up)

        # image-level residual: correct a bicubic upsample rather than
        # hallucinate the image from scratch
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return out + base


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick self-test: forward pass on a dummy batch, print param count
    m = RestorationNet()
    n = count_parameters(m)
    print(f"RestorationNet parameters: {n:,} ({n / 1e6:.2f}M)")
    x = torch.randn(2, 1, 128, 128)
    y = m(x)
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (2, 1, 256, 256), "unexpected output shape"
    print("self-test passed")
