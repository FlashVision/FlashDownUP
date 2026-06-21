"""
Lossy downsampling and upsampling operators.

These operators change spatial resolution with potential information loss:

Downsamplers:
- BilinearDown: Bilinear interpolation (anti-aliased resize)
- BicubicDown: Bicubic interpolation (sharper than bilinear)
- NearestDown: Nearest-neighbor (fastest, blocky)
- StridedConvDown: Learnable strided convolution
- MaxPoolDown: Max pooling (retains strongest activations)
- AvgPoolDown: Average pooling (retains mean signal)

Upsamplers:
- BilinearUp: Bilinear interpolation
- BicubicUp: Bicubic interpolation
- NearestUp: Nearest-neighbor
- TransposeConvUp: Learnable transposed convolution
- SubPixelUp: Sub-pixel convolution (learned PixelShuffle)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdownup.registry import register_down, register_up


# =============================================================================
# Lossy Downsamplers
# =============================================================================


@register_down("bilinear")
class BilinearDown(nn.Module):
    """
    Bilinear interpolation downsampling.

    Smooth anti-aliased resize. Good general-purpose lossy downsampler.
    Input:  (B, C, H, W)
    Output: (B, C, H // scale, W // scale)
    """

    def __init__(self, scale: int = 2, align_corners: bool = False):
        super().__init__()
        self.scale = scale
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h, target_w = H // self.scale, W // self.scale
        return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=self.align_corners)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, align_corners={self.align_corners}"


@register_down("bicubic")
class BicubicDown(nn.Module):
    """
    Bicubic interpolation downsampling.

    Sharper than bilinear, better for photographic content.
    Input:  (B, C, H, W)
    Output: (B, C, H // scale, W // scale)
    """

    def __init__(self, scale: int = 2, align_corners: bool = False):
        super().__init__()
        self.scale = scale
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h, target_w = H // self.scale, W // self.scale
        return F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=self.align_corners)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, align_corners={self.align_corners}"


@register_down("nearest")
class NearestDown(nn.Module):
    """
    Nearest-neighbor downsampling.

    Fastest method, no anti-aliasing. Produces blocky results but
    preserves exact pixel values at sample points.
    Input:  (B, C, H, W)
    Output: (B, C, H // scale, W // scale)
    """

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h, target_w = H // self.scale, W // self.scale
        return F.interpolate(x, size=(target_h, target_w), mode="nearest")

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


@register_down("strided_conv")
class StridedConvDown(nn.Module):
    """
    Learnable strided convolution downsampling.

    Uses a convolutional layer with stride to learn the optimal downsampling.
    Can be trained end-to-end. Preserves channel count by default.
    Input:  (B, C_in, H, W)
    Output: (B, C_out, H // scale, W // scale)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = None, scale: int = 2, kernel_size: int = 3):
        super().__init__()
        out_channels = out_channels or in_channels
        self.scale = scale
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=scale, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


@register_down("maxpool")
class MaxPoolDown(nn.Module):
    """
    Max pooling downsampling.

    Retains strongest activation in each pooling window.
    Good for preserving edges and salient features.
    Input:  (B, C, H, W)
    Output: (B, C, H // scale, W // scale)
    """

    def __init__(self, scale: int = 2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=scale, stride=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)

    def extra_repr(self) -> str:
        return f"scale={self.pool.kernel_size}"


@register_down("avgpool")
class AvgPoolDown(nn.Module):
    """
    Average pooling downsampling.

    Computes mean over each pooling window. Smooth but loses high-frequency detail.
    Input:  (B, C, H, W)
    Output: (B, C, H // scale, W // scale)
    """

    def __init__(self, scale: int = 2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=scale, stride=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)

    def extra_repr(self) -> str:
        return f"scale={self.pool.kernel_size}"


# =============================================================================
# Lossy Upsamplers
# =============================================================================


@register_up("bilinear")
class BilinearUp(nn.Module):
    """
    Bilinear interpolation upsampling.

    Standard smooth upsampling. No learnable parameters.
    Input:  (B, C, H, W)
    Output: (B, C, H * scale, W * scale)
    """

    def __init__(self, scale: int = 2, align_corners: bool = False):
        super().__init__()
        self.scale = scale
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=self.align_corners)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, align_corners={self.align_corners}"


@register_up("bicubic")
class BicubicUp(nn.Module):
    """
    Bicubic interpolation upsampling.

    Sharper than bilinear. Good for photographic upscaling.
    Input:  (B, C, H, W)
    Output: (B, C, H * scale, W * scale)
    """

    def __init__(self, scale: int = 2, align_corners: bool = False):
        super().__init__()
        self.scale = scale
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=self.align_corners)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, align_corners={self.align_corners}"


@register_up("nearest")
class NearestUp(nn.Module):
    """
    Nearest-neighbor upsampling.

    Fastest upsampler. Duplicates pixels — blocky but no new values introduced.
    Input:  (B, C, H, W)
    Output: (B, C, H * scale, W * scale)
    """

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="nearest")

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


@register_up("transpose_conv")
class TransposeConvUp(nn.Module):
    """
    Learnable transposed (deconvolution) upsampling.

    Uses transposed convolution to learn upsampling weights.
    Can produce checkerboard artifacts if not initialized carefully.
    Input:  (B, C_in, H, W)
    Output: (B, C_out, H * scale, W * scale)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = None, scale: int = 2, kernel_size: int = 4):
        super().__init__()
        out_channels = out_channels or in_channels
        self.scale = scale
        padding = (kernel_size - scale) // 2
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=scale, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.deconv(x))

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


@register_up("subpixel")
class SubPixelUp(nn.Module):
    """
    Sub-pixel convolution upsampling (learned PixelShuffle).

    Expands channels by scale^2 with a conv, then rearranges to spatial.
    Avoids checkerboard artifacts common with transposed convolution.
    Used in ESPCN and many super-resolution architectures.

    Input:  (B, C_in, H, W)
    Output: (B, C_out, H * scale, W * scale)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = None, scale: int = 2):
        super().__init__()
        out_channels = out_channels or in_channels
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, out_channels * scale * scale, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * scale * scale)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.bn(self.conv(x)))

    def extra_repr(self) -> str:
        return f"scale={self.scale}"
