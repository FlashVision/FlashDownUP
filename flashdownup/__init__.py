"""
FlashDownUP - Lossless, Lossy, and Feature-Preserving Image Downsampling/Upsampling.

Provides unified API for spatial resolution manipulation:
- Lossless: S2D (Space-to-Depth), D2S (Depth-to-Space), DWT (Haar wavelet)
- Lossy: Bilinear, Bicubic, Nearest, Strided Conv, Pooling-based methods
- Fractional/Dual-Branch: Feature-preserving multi-branch fusion operators
"""

__version__ = "1.1.0"

from flashdownup.ops.lossless import SpaceToDepth, DepthToSpace, DWTHaarDown, IDWTHaarUp
from flashdownup.ops.lossy import (
    BilinearDown,
    BicubicDown,
    NearestDown,
    StridedConvDown,
    MaxPoolDown,
    AvgPoolDown,
    BilinearUp,
    BicubicUp,
    NearestUp,
    TransposeConvUp,
    SubPixelUp,
)
from flashdownup.ops.fractional import (
    FractionalDualBranchDown,
    DualBranchAttentionDown,
    TriBranchDown,
    HWDDualPathDown,
    SPDConvDown,
    ConvResizeDown,
    BilinearConvDualDown,
    BilinearConvDeepDualDown,
    FractionalDualBranchUp,
    DualBranchAttentionUp,
    DySampleUp,
    CARAFELiteUp,
    ResizeConvUp,
    BilinearDeconvDualUp,
    # Advanced techniques
    BlurPoolDown,
    MaxBlurPoolDown,
    GatedConvDown,
    FrequencySplitDown,
    DeformableConvDown,
    SqueezeExciteDown,
    ProgressiveDualDown,
    MultiKernelDown,
    GatedDeconvUp,
    ProgressiveResidualUp,
    FrequencyReconstructUp,
    MultiKernelUp,
)
from flashdownup.core import FlashDown, FlashUp

__all__ = [
    # Core API
    "FlashDown",
    "FlashUp",
    # Lossless
    "SpaceToDepth",
    "DepthToSpace",
    "DWTHaarDown",
    "IDWTHaarUp",
    # Lossy
    "BilinearDown",
    "BicubicDown",
    "NearestDown",
    "StridedConvDown",
    "MaxPoolDown",
    "AvgPoolDown",
    "BilinearUp",
    "BicubicUp",
    "NearestUp",
    "TransposeConvUp",
    "SubPixelUp",
    # Fractional / Dual-Branch (Feature-Preserving)
    "FractionalDualBranchDown",
    "DualBranchAttentionDown",
    "TriBranchDown",
    "HWDDualPathDown",
    "SPDConvDown",
    "ConvResizeDown",
    "BilinearConvDualDown",
    "BilinearConvDeepDualDown",
    "FractionalDualBranchUp",
    "DualBranchAttentionUp",
    "DySampleUp",
    "CARAFELiteUp",
    "ResizeConvUp",
    "BilinearDeconvDualUp",
    # Advanced Techniques
    "BlurPoolDown",
    "MaxBlurPoolDown",
    "GatedConvDown",
    "FrequencySplitDown",
    "DeformableConvDown",
    "SqueezeExciteDown",
    "ProgressiveDualDown",
    "MultiKernelDown",
    "GatedDeconvUp",
    "ProgressiveResidualUp",
    "FrequencyReconstructUp",
    "MultiKernelUp",
]
