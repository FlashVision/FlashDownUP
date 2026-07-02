"""
Compare Fractional Dual-Branch Downsample/Upsample Approaches.

This example demonstrates the feature preservation advantages of dual-branch
and advanced resampling methods over single-method interpolation.

Key insight: Bilinear smooths edges while Nearest aliases gradients.
By fusing BOTH, we retain complementary information from each.

Methods compared:
  1. Bilinear only (baseline - smooth but loses high-freq)
  2. Nearest only (baseline - sharp but aliases)
  3. Dual-Branch (Bilinear + Nearest learned fusion)
  4. Dual-Branch Attention (per-pixel adaptive weighting)
  5. Tri-Branch (Bilinear + Nearest + Area with residual)
  6. HWD-MP Dual Path (Haar Wavelet + MaxPool)
  7. SPD-Conv (zero-loss Space-to-Depth + Conv)
  8. Conv-Resize (Learned pre-filter + resize)
  9. DySample (Dynamic point-sampling, ICCV 2023)
  10. CARAFE-Lite (Content-aware reassembly)
"""

import torch
import time
from flashdownup import FlashDown, FlashUp
from flashdownup.utils import count_params, measure_reconstruction_error


def benchmark_downsamplers():
    """Compare all downsampling methods on feature preservation."""
    print("=" * 80)
    print("DOWNSAMPLING COMPARISON — Feature Preservation Analysis")
    print("=" * 80)

    x = torch.randn(1, 64, 128, 128)
    print(f"\nInput shape: {list(x.shape)} (B=1, C=64, H=128, W=128)")
    print(f"Target: 2x downsample → (1, 64, 64, 64)\n")

    methods = {
        "bilinear (baseline)": {"method": "bilinear", "kwargs": {"scale": 2}},
        "nearest (baseline)": {"method": "nearest", "kwargs": {"scale": 2}},
        "avgpool (baseline)": {"method": "avgpool", "kwargs": {"scale": 2}},
        "dual_branch": {"method": "dual_branch", "kwargs": {"in_channels": 64, "scale_factor": 2.0}},
        "dual_branch_attn": {"method": "dual_branch_attn", "kwargs": {"in_channels": 64, "scale_factor": 2.0}},
        "tri_branch": {"method": "tri_branch", "kwargs": {"in_channels": 64, "scale_factor": 2.0}},
        "hwd_dual_path": {"method": "hwd_dual_path", "kwargs": {"in_channels": 64}},
        "spd_conv": {"method": "spd_conv", "kwargs": {"in_channels": 64, "scale": 2}},
        "conv_resize": {"method": "conv_resize", "kwargs": {"in_channels": 64, "scale_factor": 2.0}},
    }

    print(f"{'Method':<25} {'Output Shape':<20} {'Params':<10} {'Time (ms)':<12} {'Std Dev':<10}")
    print("-" * 80)

    for name, cfg in methods.items():
        op = FlashDown(cfg["method"], **cfg["kwargs"])
        params = count_params(op)

        # Warmup
        with torch.no_grad():
            _ = op(x)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(50):
                start = time.perf_counter()
                y = op(x)
                times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        # Measure output variance (higher = more feature diversity preserved)
        std_dev = y.std().item()

        print(f"{name:<25} {str(list(y.shape)):<20} {params:<10} {avg_time:<12.3f} {std_dev:<10.4f}")


def benchmark_upsamplers():
    """Compare all upsampling methods."""
    print("\n" + "=" * 80)
    print("UPSAMPLING COMPARISON — Feature Preservation Analysis")
    print("=" * 80)

    x = torch.randn(1, 64, 32, 32)
    print(f"\nInput shape: {list(x.shape)} (B=1, C=64, H=32, W=32)")
    print(f"Target: 2x upsample → (1, 64, 64, 64)\n")

    methods = {
        "bilinear (baseline)": {"method": "bilinear", "kwargs": {"scale": 2}},
        "nearest (baseline)": {"method": "nearest", "kwargs": {"scale": 2}},
        "dual_branch": {"method": "dual_branch", "kwargs": {"in_channels": 64, "scale_factor": 2.0}},
        "dual_branch_attn": {"method": "dual_branch_attn", "kwargs": {"in_channels": 64, "scale_factor": 2.0}},
        "dysample (LP)": {"method": "dysample", "kwargs": {"in_channels": 64, "scale": 2, "style": "lp"}},
        "dysample (PL)": {"method": "dysample", "kwargs": {"in_channels": 64, "scale": 2, "style": "pl"}},
        "carafe_lite": {"method": "carafe_lite", "kwargs": {"in_channels": 64, "scale": 2}},
        "resize_conv": {"method": "resize_conv", "kwargs": {"in_channels": 64, "scale_factor": 2.0}},
        "subpixel": {"method": "subpixel", "kwargs": {"in_channels": 64, "scale": 2}},
    }

    print(f"{'Method':<25} {'Output Shape':<20} {'Params':<10} {'Time (ms)':<12} {'Std Dev':<10}")
    print("-" * 80)

    for name, cfg in methods.items():
        op = FlashUp(cfg["method"], **cfg["kwargs"])
        params = count_params(op)

        with torch.no_grad():
            _ = op(x)

        times = []
        with torch.no_grad():
            for _ in range(50):
                start = time.perf_counter()
                y = op(x)
                times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        std_dev = y.std().item()

        print(f"{name:<25} {str(list(y.shape)):<20} {params:<10} {avg_time:<12.3f} {std_dev:<10.4f}")


def demo_fractional_scaling():
    """Demonstrate non-integer scale factor support."""
    print("\n" + "=" * 80)
    print("FRACTIONAL SCALING DEMO — Non-Integer Scale Factors")
    print("=" * 80)

    x = torch.randn(1, 3, 1080, 1920)  # Full HD
    print(f"\nInput: 1080p ({list(x.shape)})")

    scales = [
        (1.5, "720p"),    # 1080 / 1.5 = 720
        (2.0, "540p"),    # 1080 / 2.0 = 540
        (2.25, "480p"),   # 1080 / 2.25 = 480
        (4.0, "270p"),    # 1080 / 4.0 = 270
    ]

    for scale, target_name in scales:
        down = FlashDown("dual_branch", in_channels=3, scale_factor=scale)
        with torch.no_grad():
            y = down(x)
        print(f"  scale_factor={scale:<5} → {target_name:<6} {list(y.shape)}")


def reconstruction_quality_comparison():
    """Compare down→up reconstruction quality across methods."""
    print("\n" + "=" * 80)
    print("RECONSTRUCTION QUALITY — Down(2x) → Up(2x) Error Analysis")
    print("=" * 80)

    torch.manual_seed(42)
    x = torch.randn(1, 32, 64, 64)
    print(f"\nInput shape: {list(x.shape)}")
    print(f"Pipeline: Downsample 2x → Upsample 2x → Compare with original\n")

    pairs = [
        ("bilinear → bilinear", FlashDown("bilinear", scale=2), FlashUp("bilinear", scale=2)),
        ("nearest → nearest", FlashDown("nearest", scale=2), FlashUp("nearest", scale=2)),
        ("avgpool → bilinear", FlashDown("avgpool", scale=2), FlashUp("bilinear", scale=2)),
        (
            "dual_branch → dual_branch",
            FlashDown("dual_branch", in_channels=32, scale_factor=2.0),
            FlashUp("dual_branch", in_channels=32, scale_factor=2.0),
        ),
        (
            "dual_branch_attn → dual_branch_attn",
            FlashDown("dual_branch_attn", in_channels=32, scale_factor=2.0),
            FlashUp("dual_branch_attn", in_channels=32, scale_factor=2.0),
        ),
        (
            "hwd_dual_path → bilinear",
            FlashDown("hwd_dual_path", in_channels=32),
            FlashUp("bilinear", scale=2),
        ),
        (
            "spd_conv → subpixel",
            FlashDown("spd_conv", in_channels=32, scale=2),
            FlashUp("subpixel", in_channels=32, scale=2),
        ),
    ]

    print(f"{'Method Pair':<40} {'MSE':<12} {'PSNR (dB)':<12}")
    print("-" * 65)

    for name, down_op, up_op in pairs:
        result = measure_reconstruction_error(down_op, up_op, x)
        if "error" in result:
            print(f"{name:<40} {result['error']}")
        else:
            print(f"{name:<40} {result['mse']:<12.6f} {result['psnr_db']:<12.2f}")


if __name__ == "__main__":
    benchmark_downsamplers()
    benchmark_upsamplers()
    demo_fractional_scaling()
    reconstruction_quality_comparison()

    print("\n" + "=" * 80)
    print("SUMMARY OF APPROACHES")
    print("=" * 80)
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ APPROACH                 │ FEATURE LOSS │ LEARNABLE │ FRACTIONAL │ SPEED    │
├──────────────────────────┼──────────────┼───────────┼────────────┼──────────┤
│ Bilinear (baseline)      │ High (smooth)│ No        │ Yes        │ Fastest  │
│ Nearest (baseline)       │ High (alias) │ No        │ Yes        │ Fastest  │
│ Dual-Branch (ours)       │ LOW          │ Yes       │ Yes        │ Fast     │
│ Dual-Branch + Attention  │ LOWEST       │ Yes       │ Yes        │ Medium   │
│ Tri-Branch               │ VERY LOW     │ Yes       │ Yes        │ Medium   │
│ HWD + MaxPool Dual Path  │ VERY LOW     │ Yes       │ No (2x)   │ Fast     │
│ SPD-Conv                 │ ZERO (lossless)│ Yes     │ No (int)   │ Fast     │
│ Conv-Resize              │ Low          │ Yes       │ Yes        │ Medium   │
│ DySample                 │ VERY LOW     │ Yes       │ No (int)   │ Fast     │
│ CARAFE-Lite              │ LOW          │ Yes       │ No (int)   │ Slower   │
└─────────────────────────────────────────────────────────────────────────────┘

RECOMMENDED:
  • Best overall: dual_branch_attn (attention-gated bilinear+nearest fusion)
  • Best for integer 2x: hwd_dual_path (wavelet preserves ALL frequencies)
  • Best upsampler: dysample (SOTA, ultra-lightweight, no CUDA deps)
  • Zero info loss: spd_conv (space-to-depth, guaranteed lossless spatial→channel)
  • Best for fractional: dual_branch or conv_resize (supports 1080p→720p etc.)
""")
