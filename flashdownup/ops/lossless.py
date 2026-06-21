"""
Lossless downsampling and upsampling operators.

These operators perform spatial resolution changes without information loss:
- SpaceToDepth (S2D): Rearranges spatial blocks into channel dimension (PixelUnshuffle)
- DepthToSpace (D2S): Rearranges channel dimension into spatial blocks (PixelShuffle)
- DWTHaarDown: Haar wavelet decomposition (lossless 2x downsample into 4 subbands)
- IDWTHaarUp: Inverse Haar wavelet reconstruction (lossless 2x upsample from 4 subbands)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdownup.registry import register_down, register_up


@register_down("s2d")
class SpaceToDepth(nn.Module):
    """
    Space-to-Depth (S2D) / PixelUnshuffle.

    Rearranges spatial data into depth (channel) dimension.
    Input:  (B, C, H, W)
    Output: (B, C * scale^2, H // scale, W // scale)

    This is a lossless operation - no information is discarded.
    Commonly used as a learnable-free downsampling stem in detection/segmentation models.
    """

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(x, self.scale)

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


@register_up("d2s")
class DepthToSpace(nn.Module):
    """
    Depth-to-Space (D2S) / PixelShuffle.

    Rearranges depth (channel) data into spatial dimensions.
    Input:  (B, C * scale^2, H, W)
    Output: (B, C, H * scale, W * scale)

    This is a lossless operation - inverse of SpaceToDepth.
    Commonly used in super-resolution and generative models.
    """

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(x, self.scale)

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


@register_down("dwt_haar")
class DWTHaarDown(nn.Module):
    """
    Discrete Wavelet Transform (Haar) Downsampling.

    Applies Haar wavelet decomposition for lossless 2x spatial downsampling.
    Decomposes input into 4 subbands: LL (approx), LH (horizontal detail),
    HL (vertical detail), HH (diagonal detail).

    Input:  (B, C, H, W)
    Output: (B, C * 4, H // 2, W // 2)

    The LL subband is the low-frequency approximation (like a proper downsample).
    LH, HL, HH capture high-frequency details that enable perfect reconstruction.
    """

    def __init__(self):
        super().__init__()
        # Haar wavelet filters (normalized)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        # Stack into (4, 1, 2, 2) filter bank
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Apply per-channel: reshape to (B*C, 1, H, W), convolve, reshape back
        x_flat = x.reshape(B * C, 1, H, W)
        # Stride-2 convolution with Haar filters produces (B*C, 4, H//2, W//2)
        out = F.conv2d(x_flat, self.filters, stride=2)
        # Reshape to (B, C*4, H//2, W//2)
        return out.reshape(B, C * 4, H // 2, W // 2)

    def extra_repr(self) -> str:
        return "wavelet=haar, subbands=4 (LL, LH, HL, HH)"


@register_up("idwt_haar")
class IDWTHaarUp(nn.Module):
    """
    Inverse Discrete Wavelet Transform (Haar) Upsampling.

    Reconstructs spatial resolution from Haar wavelet subbands.
    Perfect reconstruction from DWTHaarDown output.

    Input:  (B, C * 4, H, W) — 4 subbands per channel
    Output: (B, C, H * 2, W * 2)
    """

    def __init__(self):
        super().__init__()
        # Inverse Haar filters for transposed convolution
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        # For conv_transpose2d: weight shape is (in_channels, out_channels, kH, kW)
        # We have 4 input channels (subbands) -> 1 output channel per group
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)  # (4, 1, 2, 2)
        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C4, H, W = x.shape
        assert C4 % 4 == 0, f"Channel dim must be divisible by 4, got {C4}"
        C = C4 // 4

        # Reshape to (B*C, 4, H, W) — treat each group of 4 subbands
        x_flat = x.reshape(B * C, 4, H, W)
        # Transposed convolution with groups: (B*C, 4, H, W) -> (B*C, 4, H*2, W*2)
        # Then sum across the 4 subbands to get (B*C, 1, H*2, W*2)
        # Use grouped conv_transpose2d then sum, or just do it manually
        out = torch.zeros(B * C, 1, H * 2, W * 2, device=x.device, dtype=x.dtype)
        for i in range(4):
            out += F.conv_transpose2d(x_flat[:, i:i+1, :, :], self.filters[i:i+1, :, :, :], stride=2)
        # Reshape to (B, C, H*2, W*2)
        return out.reshape(B, C, H * 2, W * 2)

    def extra_repr(self) -> str:
        return "wavelet=haar, reconstruction=perfect"
