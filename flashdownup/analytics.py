"""Analytics and profiling for FlashDownUP operators."""

import time
from typing import Optional

import torch
import torch.nn as nn


def profile_operator(
    op: nn.Module,
    input_shape: tuple = (1, 3, 256, 256),
    device: str = "cpu",
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """
    Profile a downsampling/upsampling operator.

    Returns timing, memory, and throughput statistics.
    """
    op = op.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = op(x)

    if device == "cuda":
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            y = op(x)
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)

    timings_ms = [t * 1000 for t in timings]
    params = sum(p.numel() for p in op.parameters())

    return {
        "input_shape": list(input_shape),
        "output_shape": list(y.shape),
        "mean_ms": sum(timings_ms) / len(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "std_ms": (sum((t - sum(timings_ms) / len(timings_ms)) ** 2 for t in timings_ms) / len(timings_ms)) ** 0.5,
        "params": params,
        "device": device,
        "iterations": iterations,
    }


def compare_methods(
    methods: list,
    direction: str = "down",
    input_shape: tuple = (1, 3, 256, 256),
    device: str = "cpu",
    scale: int = 2,
) -> list:
    """
    Compare multiple down/up methods side by side.

    Args:
        methods: List of method names to compare.
        direction: "down" or "up".
        input_shape: Input tensor shape.
        device: Device to profile on.
        scale: Scale factor.

    Returns:
        List of profiling dicts for each method.
    """
    from flashdownup.core import FlashDown, FlashUp

    results = []
    for method in methods:
        kwargs = {"scale": scale}
        C = input_shape[1]
        if method in ("strided_conv", "transpose_conv", "subpixel"):
            kwargs["in_channels"] = C

        if direction == "down":
            op = FlashDown(method, **kwargs)
        else:
            op = FlashUp(method, **kwargs)

        result = profile_operator(op, input_shape=input_shape, device=device)
        result["method"] = method
        result["direction"] = direction
        results.append(result)

    return results
