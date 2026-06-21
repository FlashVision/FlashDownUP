# FlashDownUP

**Lossless and Lossy Image Downsampling/Upsampling Operators for PyTorch**

Part of the [FlashVision](https://github.com/FlashVision) ecosystem.

## Overview

FlashDownUP provides a unified API for spatial resolution manipulation in deep learning pipelines. It implements both **lossless** (information-preserving) and **lossy** (interpolation-based) operators as drop-in `nn.Module` components.

## Methods

### Downsampling (`FlashDown`)

| Method | Type | Output Shape | Learnable | Description |
|--------|------|-------------|-----------|-------------|
| `s2d` | Lossless | (B, C×s², H/s, W/s) | No | Space-to-Depth / PixelUnshuffle |
| `dwt_haar` | Lossless | (B, C×4, H/2, W/2) | No | Haar wavelet decomposition (LL, LH, HL, HH subbands) |
| `bilinear` | Lossy | (B, C, H/s, W/s) | No | Bilinear interpolation |
| `bicubic` | Lossy | (B, C, H/s, W/s) | No | Bicubic interpolation |
| `nearest` | Lossy | (B, C, H/s, W/s) | No | Nearest-neighbor |
| `strided_conv` | Lossy | (B, C, H/s, W/s) | Yes | Learnable strided convolution |
| `maxpool` | Lossy | (B, C, H/s, W/s) | No | Max pooling |
| `avgpool` | Lossy | (B, C, H/s, W/s) | No | Average pooling |

### Upsampling (`FlashUp`)

| Method | Type | Output Shape | Learnable | Description |
|--------|------|-------------|-----------|-------------|
| `d2s` | Lossless | (B, C/s², H×s, W×s) | No | Depth-to-Space / PixelShuffle |
| `idwt_haar` | Lossless | (B, C/4, H×2, W×2) | No | Inverse Haar wavelet (perfect reconstruction) |
| `bilinear` | Lossy | (B, C, H×s, W×s) | No | Bilinear interpolation |
| `bicubic` | Lossy | (B, C, H×s, W×s) | No | Bicubic interpolation |
| `nearest` | Lossy | (B, C, H×s, W×s) | No | Nearest-neighbor |
| `transpose_conv` | Lossy | (B, C, H×s, W×s) | Yes | Transposed convolution |
| `subpixel` | Lossy | (B, C, H×s, W×s) | Yes | Sub-pixel conv (ESPCN-style) |

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from flashdownup import FlashDown, FlashUp

x = torch.randn(1, 3, 256, 256)

# Lossless: Space-to-Depth (S2D)
down = FlashDown("s2d", scale=2)
encoded = down(x)  # (1, 12, 128, 128) — no info lost

up = FlashUp("d2s", scale=2)
reconstructed = up(encoded)  # (1, 3, 256, 256) — perfect
assert torch.allclose(x, reconstructed)

# Lossless: Haar Wavelet (DWT)
down = FlashDown("dwt_haar")
subbands = down(x)  # (1, 12, 128, 128) — LL, LH, HL, HH

up = FlashUp("idwt_haar")
reconstructed = up(subbands)  # (1, 3, 256, 256) — perfect
assert torch.allclose(x, reconstructed, atol=1e-5)

# Lossy: Bilinear
down = FlashDown("bilinear", scale=4)
small = down(x)  # (1, 3, 64, 64) — lossy resize
```

## Key Concepts

### Lossless vs Lossy

**Lossless** operators preserve all information by expanding the channel dimension:
- **S2D**: Rearranges s×s spatial blocks into s² channels. Used as backbone stems (YOLOv5 Focus).
- **DWT Haar**: Decomposes into frequency subbands. LL = approximation, LH/HL/HH = details.

**Lossy** operators reduce spatial size while keeping channels constant:
- Interpolation-based (bilinear, bicubic, nearest) — fast, no parameters
- Learnable (strided_conv, subpixel) — trainable, task-adaptive

### When to Use What

| Use Case | Recommended Method |
|----------|-------------------|
| Backbone stem (preserve small objects) | `s2d` |
| Feature pyramid downsample | `strided_conv` or `avgpool` |
| Frequency-domain processing | `dwt_haar` |
| Super-resolution upsampling | `subpixel` |
| Decoder upsampling (segmentation) | `bilinear` or `transpose_conv` |
| Fastest possible resize | `nearest` |

## CLI

```bash
# List all methods
flashdownup list

# Benchmark a method
flashdownup bench s2d --size 512 --channels 3
flashdownup bench dwt_haar --size 256
flashdownup bench bilinear --direction up --size 64 --scale 4
```

## Architecture

```
flashdownup/
├── __init__.py         # Public API exports
├── core.py             # FlashDown, FlashUp unified interface
├── registry.py         # Method registration system
├── ops/
│   ├── lossless.py     # S2D, D2S, DWT Haar, IDWT Haar
│   └── lossy.py        # Bilinear, Bicubic, Nearest, StridedConv, Pools, TransposeConv, SubPixel
├── utils.py            # Reconstruction error, compression ratio
├── analytics.py        # Profiling and method comparison
└── cli.py              # Command-line interface
```

## License

MIT
