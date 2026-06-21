"""Utility functions for FlashDownUP."""

import torch
import torch.nn as nn


def count_params(module: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def measure_reconstruction_error(
    down_op: nn.Module, up_op: nn.Module, x: torch.Tensor
) -> dict:
    """
    Measure reconstruction quality for a down/up pair.

    Returns dict with PSNR, MSE, and whether reconstruction is perfect.
    """
    with torch.no_grad():
        encoded = down_op(x)
        reconstructed = up_op(encoded)

    if reconstructed.shape != x.shape:
        return {"error": f"Shape mismatch: {x.shape} vs {reconstructed.shape}", "perfect": False}

    mse = torch.mean((x - reconstructed) ** 2).item()
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()

    return {
        "mse": mse,
        "psnr_db": psnr,
        "perfect": mse < 1e-10,
        "input_shape": list(x.shape),
        "encoded_shape": list(encoded.shape),
        "reconstructed_shape": list(reconstructed.shape),
    }


def compute_compression_ratio(x: torch.Tensor, encoded: torch.Tensor) -> float:
    """Compute spatial compression ratio (input pixels / encoded pixels)."""
    in_pixels = x.shape[2] * x.shape[3]
    enc_pixels = encoded.shape[2] * encoded.shape[3]
    return in_pixels / enc_pixels
