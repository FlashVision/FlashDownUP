"""
Core unified API: FlashDown and FlashUp.

Provides a single entry point for selecting any downsampling or upsampling
method by name, with consistent interface.
"""

import torch
import torch.nn as nn

from flashdownup.registry import get_downsampler, get_upsampler, list_downsamplers, list_upsamplers


class FlashDown(nn.Module):
    """
    Unified downsampling operator.

    Supports lossless, lossy, and feature-preserving methods via a single interface.

    Lossless methods (no information loss, channel expansion):
        - "s2d": Space-to-Depth / PixelUnshuffle. Output: (B, C*scale^2, H//scale, W//scale)
        - "dwt_haar": Haar wavelet decomposition. Output: (B, C*4, H//2, W//2)

    Lossy methods (information loss, channels preserved):
        - "bilinear": Bilinear interpolation
        - "bicubic": Bicubic interpolation
        - "nearest": Nearest-neighbor
        - "strided_conv": Learnable strided convolution
        - "maxpool": Max pooling
        - "avgpool": Average pooling

    Fractional / Dual-Branch (feature-preserving, learnable fusion):
        - "dual_branch": Bilinear + Nearest with learned 1x1 conv fusion
        - "dual_branch_attn": Bilinear + Nearest with spatial attention gate
        - "tri_branch": Bilinear + Nearest + Area with residual fusion
        - "hwd_dual_path": Haar Wavelet + MaxPool with channel attention gate
        - "spd_conv": Space-to-Depth + non-strided conv (zero spatial info loss)
        - "conv_resize": Learned pre-filter + bilinear resize (fractional scales)

    Args:
        method: Name of the downsampling method.
        **kwargs: Method-specific arguments (scale, in_channels, scale_factor, etc.)

    Example:
        >>> down = FlashDown("s2d", scale=2)
        >>> x = torch.randn(1, 3, 64, 64)
        >>> y = down(x)  # (1, 12, 32, 32)

        >>> down = FlashDown("dual_branch", in_channels=64, scale_factor=2.0)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> y = down(x)  # (1, 64, 16, 16) — fused bilinear+nearest

        >>> down = FlashDown("hwd_dual_path", in_channels=64)
        >>> y = down(x)  # (1, 64, 16, 16) — HWD+MaxPool fusion
    """

    def __init__(self, method: str = "s2d", **kwargs):
        super().__init__()
        self.method_name = method
        self.op = get_downsampler(method, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

    @staticmethod
    def available_methods() -> list:
        """Return list of all registered downsampling methods."""
        return list_downsamplers()

    def extra_repr(self) -> str:
        return f"method={self.method_name}"


class FlashUp(nn.Module):
    """
    Unified upsampling operator.

    Supports lossless, lossy, and feature-preserving methods via a single interface.

    Lossless methods (perfect reconstruction from encoded input):
        - "d2s": Depth-to-Space / PixelShuffle. Output: (B, C//scale^2, H*scale, W*scale)
        - "idwt_haar": Inverse Haar wavelet. Output: (B, C//4, H*2, W*2)

    Lossy methods (interpolation-based, no perfect reconstruction):
        - "bilinear": Bilinear interpolation
        - "bicubic": Bicubic interpolation
        - "nearest": Nearest-neighbor
        - "transpose_conv": Learnable transposed convolution
        - "subpixel": Sub-pixel convolution (learned PixelShuffle)

    Fractional / Dual-Branch (feature-preserving, learnable fusion):
        - "dual_branch": Bilinear + Nearest with learned fusion
        - "dual_branch_attn": Bilinear + Nearest with spatial attention gate
        - "dysample": Dynamic point-sampling upsampler (ICCV 2023, SOTA)
        - "carafe_lite": Content-aware reassembly with adaptive kernels
        - "resize_conv": Bilinear resize + learned refinement conv

    Args:
        method: Name of the upsampling method.
        **kwargs: Method-specific arguments (scale, in_channels, scale_factor, etc.)

    Example:
        >>> up = FlashUp("d2s", scale=2)
        >>> x = torch.randn(1, 12, 32, 32)
        >>> y = up(x)  # (1, 3, 64, 64)

        >>> up = FlashUp("dual_branch", in_channels=64, scale_factor=2.0)
        >>> x = torch.randn(1, 64, 16, 16)
        >>> y = up(x)  # (1, 64, 32, 32) — fused bilinear+nearest

        >>> up = FlashUp("dysample", in_channels=64, scale=2)
        >>> y = up(x)  # (1, 64, 32, 32) — dynamic point sampling
    """

    def __init__(self, method: str = "d2s", **kwargs):
        super().__init__()
        self.method_name = method
        self.op = get_upsampler(method, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

    @staticmethod
    def available_methods() -> list:
        """Return list of all registered upsampling methods."""
        return list_upsamplers()

    def extra_repr(self) -> str:
        return f"method={self.method_name}"
