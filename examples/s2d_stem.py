"""Example: Use S2D/D2S as stem in a detection backbone."""

import torch
import torch.nn as nn
from flashdownup import SpaceToDepth, DepthToSpace


class S2DStem(nn.Module):
    """
    Space-to-Depth stem for efficient input processing.

    Instead of a strided conv that loses information, S2D rearranges
    spatial pixels into channels, then a 1x1 conv projects to desired dims.
    This is used in YOLOv5 Focus, some EfficientDet variants, etc.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64, scale: int = 2):
        super().__init__()
        self.s2d = SpaceToDepth(scale=scale)
        expanded_channels = in_channels * scale * scale
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.s2d(x))


def main():
    stem = S2DStem(in_channels=3, out_channels=64, scale=2)
    x = torch.randn(1, 3, 640, 640)

    y = stem(x)
    print(f"S2D Stem:")
    print(f"  Input:  {list(x.shape)}")
    print(f"  Output: {list(y.shape)}")
    print(f"  Params: {sum(p.numel() for p in stem.parameters()):,}")
    print()
    print("  Advantage over strided conv:")
    print("  - Zero information loss in the spatial rearrangement step")
    print("  - 1x1 projection is cheaper than 3x3 or 6x6 strided conv")
    print("  - Better gradient flow for small objects")


if __name__ == "__main__":
    main()
