"""CLI interface for FlashDownUP."""

import argparse
import sys

import torch


def main():
    parser = argparse.ArgumentParser(
        prog="flashdownup",
        description="FlashDownUP: Lossless & Lossy Image Downsampling/Upsampling Operators",
    )
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_parser = subparsers.add_parser("list", help="List available methods")
    list_parser.add_argument("--type", choices=["down", "up", "all"], default="all", help="Filter by type")

    # Benchmark command
    bench_parser = subparsers.add_parser("bench", help="Benchmark a method")
    bench_parser.add_argument("method", type=str, help="Method name (e.g., s2d, dwt_haar, bilinear)")
    bench_parser.add_argument("--direction", choices=["down", "up"], default="down")
    bench_parser.add_argument("--size", type=int, default=256, help="Input spatial size")
    bench_parser.add_argument("--channels", type=int, default=3, help="Input channels")
    bench_parser.add_argument("--batch", type=int, default=1, help="Batch size")
    bench_parser.add_argument("--scale", type=int, default=2, help="Scale factor")
    bench_parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    bench_parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")

    args = parser.parse_args()

    if args.command == "list":
        _cmd_list(args)
    elif args.command == "bench":
        _cmd_bench(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_list(args):
    from flashdownup.registry import list_downsamplers, list_upsamplers

    if args.type in ("down", "all"):
        print("Downsamplers:")
        print("  Lossless: s2d, dwt_haar")
        print("  Lossy:    bilinear, bicubic, nearest, strided_conv, maxpool, avgpool")
        print()

    if args.type in ("up", "all"):
        print("Upsamplers:")
        print("  Lossless: d2s, idwt_haar")
        print("  Lossy:    bilinear, bicubic, nearest, transpose_conv, subpixel")


def _cmd_bench(args):
    import time
    from flashdownup.core import FlashDown, FlashUp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    C, H, W = args.channels, args.size, args.size

    kwargs = {"scale": args.scale}
    if args.method in ("strided_conv", "transpose_conv", "subpixel"):
        kwargs["in_channels"] = C

    if args.direction == "down":
        model = FlashDown(args.method, **kwargs).to(device).eval()
        x = torch.randn(args.batch, C, H, W, device=device)
    else:
        if args.method in ("d2s",):
            x = torch.randn(args.batch, C * args.scale ** 2, H, W, device=device)
        elif args.method == "idwt_haar":
            x = torch.randn(args.batch, C * 4, H, W, device=device)
        else:
            x = torch.randn(args.batch, C, H, W, device=device)
        model = FlashUp(args.method, **kwargs).to(device).eval()

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.iters):
            y = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / args.iters * 1000

    print(f"Method:    {args.method} ({args.direction})")
    print(f"Device:    {device}")
    print(f"Input:     ({args.batch}, {x.shape[1]}, {x.shape[2]}, {x.shape[3]})")
    print(f"Output:    ({y.shape[0]}, {y.shape[1]}, {y.shape[2]}, {y.shape[3]})")
    print(f"Latency:   {elapsed:.3f} ms (avg over {args.iters} iters)")
    params = sum(p.numel() for p in model.parameters())
    print(f"Params:    {params:,}")


if __name__ == "__main__":
    main()
