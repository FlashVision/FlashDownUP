"""
FlashDownUP - Lossless and Lossy Image Downsampling/Upsampling Operators.

Provides unified API for spatial resolution manipulation:
- Lossless: S2D (Space-to-Depth), D2S (Depth-to-Space), DWT (Haar wavelet)
- Lossy: Bilinear, Bicubic, Nearest, Strided Conv, Pooling-based methods
"""

__version__ = "1.0.0"

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
from flashdownup.core import FlashDown, FlashUp

__all__ = [
    "FlashDown",
    "FlashUp",
    "SpaceToDepth",
    "DepthToSpace",
    "DWTHaarDown",
    "IDWTHaarUp",
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
]
