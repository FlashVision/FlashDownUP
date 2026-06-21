"""Example: Compare all downsampling methods."""

import torch
from flashdownup import FlashDown, FlashUp
from flashdownup.utils import measure_reconstruction_error


def main():
    x = torch.randn(1, 3, 256, 256)
    print("=" * 70)
    print("FlashDownUP: Downsampling Methods Comparison")
    print("=" * 70)
    print(f"Input shape: {list(x.shape)}")
    print()

    # Lossless methods
    print("--- LOSSLESS (perfect reconstruction) ---")
    for method, up_method in [("s2d", "d2s"), ("dwt_haar", "idwt_haar")]:
        down = FlashDown(method)
        up = FlashUp(up_method)
        y = down(x)
        result = measure_reconstruction_error(down, up, x)
        print(f"  {method:12s} | output: {list(y.shape)} | perfect: {result['perfect']} | PSNR: {result['psnr_db']:.1f} dB")

    print()
    print("--- LOSSY (information loss) ---")
    for method in ["bilinear", "bicubic", "nearest", "maxpool", "avgpool"]:
        down = FlashDown(method, scale=2)
        y = down(x)
        print(f"  {method:12s} | output: {list(y.shape)}")

    # Strided conv needs in_channels
    down = FlashDown("strided_conv", in_channels=3, scale=2)
    y = down(x)
    print(f"  {'strided_conv':12s} | output: {list(y.shape)} | learnable params: {sum(p.numel() for p in down.parameters()):,}")

    print()
    print("--- UPSAMPLING ---")
    small = torch.randn(1, 3, 64, 64)
    for method in ["bilinear", "bicubic", "nearest"]:
        up = FlashUp(method, scale=2)
        y = up(small)
        print(f"  {method:12s} | input: {list(small.shape)} -> output: {list(y.shape)}")

    for method in ["transpose_conv", "subpixel"]:
        up = FlashUp(method, in_channels=3, scale=2)
        y = up(small)
        print(f"  {method:12s} | input: {list(small.shape)} -> output: {list(y.shape)} | params: {sum(p.numel() for p in up.parameters()):,}")

    # D2S / IDWT require proper channel arrangement
    encoded = torch.randn(1, 12, 64, 64)
    up = FlashUp("d2s", scale=2)
    y = up(encoded)
    print(f"  {'d2s':12s} | input: {list(encoded.shape)} -> output: {list(y.shape)}")

    up = FlashUp("idwt_haar")
    y = up(encoded)
    print(f"  {'idwt_haar':12s} | input: {list(encoded.shape)} -> output: {list(y.shape)}")


if __name__ == "__main__":
    main()
