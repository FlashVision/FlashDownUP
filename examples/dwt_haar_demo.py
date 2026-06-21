"""Example: Use DWT Haar wavelet for lossless image downsampling."""

import torch
from flashdownup import DWTHaarDown, IDWTHaarUp


def main():
    # Simulate an RGB image
    image = torch.randn(1, 3, 128, 128)
    print(f"Original image shape: {list(image.shape)}")

    # Haar wavelet decomposition (2x downsample, lossless)
    dwt = DWTHaarDown()
    subbands = dwt(image)
    print(f"After DWT Haar: {list(subbands.shape)}")
    print(f"  -> 4 subbands per channel: LL (approx), LH (horiz), HL (vert), HH (diag)")
    print(f"  -> Spatial: {image.shape[2]}x{image.shape[3]} -> {subbands.shape[2]}x{subbands.shape[3]}")
    print(f"  -> Channels: {image.shape[1]} -> {subbands.shape[1]} (C*4)")

    # Perfect reconstruction via inverse DWT
    idwt = IDWTHaarUp()
    reconstructed = idwt(subbands)
    print(f"\nReconstructed shape: {list(reconstructed.shape)}")

    error = torch.max(torch.abs(image - reconstructed)).item()
    print(f"Max reconstruction error: {error:.2e}")
    print(f"Perfect reconstruction: {error < 1e-5}")

    # Extract just the LL (low-frequency approximation) subband
    C = image.shape[1]
    ll_band = subbands[:, 0::4, :, :]  # every 4th channel starting at 0
    print(f"\nLL subband only (low-freq approx): {list(ll_band.shape)}")
    print("  -> This is the 'proper' downsampled version")
    print("  -> LH/HL/HH carry the detail needed for perfect reconstruction")


if __name__ == "__main__":
    main()
