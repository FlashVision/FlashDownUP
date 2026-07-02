"""
Fractional and Dual-Branch Downsampling/Upsampling Operators.

Feature-preserving resampling through multi-branch fusion strategies:

Downsamplers:
- FractionalDualBranchDown: Bilinear + Nearest dual-branch with learnable merge
- HWDDualPathDown: Haar Wavelet + MaxPool dual-path (IEEE 2024)
- SPDConvDown: Space-to-Depth + non-strided conv (zero info loss)
- ConvResizeDown: Conv followed by differentiable resize (fractional factors)

Upsamplers:
- FractionalDualBranchUp: Bilinear + Nearest dual-branch with learnable merge
- DySampleUp: Dynamic point-sampling upsampler (ICCV 2023)
- CARAFELiteUp: Lightweight content-aware reassembly upsampler

Key Idea: Single interpolation methods each have distinct failure modes:
  - Bilinear: smooths high-frequency details (edges, textures)
  - Nearest: preserves sharpness but creates aliasing artifacts
  By running BOTH in parallel and learning to merge, the network retains
  complementary information from each branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdownup.registry import register_down, register_up


# =============================================================================
# Dual-Branch Fractional Downsamplers
# =============================================================================


@register_down("dual_branch")
class FractionalDualBranchDown(nn.Module):
    """
    Dual-Branch Fractional Downsampler with learnable feature fusion.

    Architecture:
        Input ──┬── Branch A (Bilinear) ──┐
                │                          ├── Concat ── Conv1x1 ── BN ── Output
                └── Branch B (Nearest)  ───┘

    Bilinear branch preserves smooth low-frequency structure.
    Nearest branch preserves sharp edges and exact pixel values.
    A learnable 1x1 conv fuses both, allowing the network to adaptively
    weight each branch per-channel for optimal feature preservation.

    Supports fractional scale factors (e.g., 1080p -> 720p = 1.5x).

    Input:  (B, C, H, W)
    Output: (B, C, H_out, W_out) where H_out = H / scale_factor
    """

    def __init__(self, in_channels: int = 3, scale_factor: float = 2.0,
                 align_corners: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h = int(H / self.scale_factor)
        target_w = int(W / self.scale_factor)

        branch_bilinear = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear",
            align_corners=self.align_corners
        )
        branch_nearest = F.interpolate(
            x, size=(target_h, target_w), mode="nearest"
        )

        fused = torch.cat([branch_bilinear, branch_nearest], dim=1)
        return self.fusion(fused)

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}, branches=[bilinear, nearest]"


@register_down("dual_branch_attn")
class DualBranchAttentionDown(nn.Module):
    """
    Dual-Branch Downsampler with channel-spatial attention fusion.

    Instead of simple concat+conv, uses learnable attention weights to
    dynamically blend bilinear and nearest branches per spatial location.
    This allows the network to use nearest-neighbor in edge regions and
    bilinear in smooth regions.

    Architecture:
        Input ──┬── Branch A (Bilinear) ──┐
                │                          ├── Attention Gate ── Weighted Sum ── Output
                └── Branch B (Nearest)  ───┘

    The attention gate predicts a per-pixel alpha map from the concatenated
    features, then blends: output = alpha * bilinear + (1-alpha) * nearest.

    Input:  (B, C, H, W)
    Output: (B, C, H_out, W_out)
    """

    def __init__(self, in_channels: int = 3, scale_factor: float = 2.0,
                 align_corners: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners

        self.attn_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h = int(H / self.scale_factor)
        target_w = int(W / self.scale_factor)

        branch_bilinear = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear",
            align_corners=self.align_corners
        )
        branch_nearest = F.interpolate(
            x, size=(target_h, target_w), mode="nearest"
        )

        concat = torch.cat([branch_bilinear, branch_nearest], dim=1)
        alpha = self.attn_gate(concat)

        return alpha * branch_bilinear + (1 - alpha) * branch_nearest

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}, fusion=attention_gate"


@register_down("tri_branch")
class TriBranchDown(nn.Module):
    """
    Tri-Branch Downsampler: Bilinear + Nearest + Area (average) merge.

    Three complementary interpolation strategies:
      - Bilinear: smooth, preserves gradients
      - Nearest: sharp, preserves exact values at sample points
      - Area: proper anti-aliased downsample (equivalent to avgpool for integer scales)

    All three are merged with a learned 1x1 convolution with residual connection.

    Input:  (B, C, H, W)
    Output: (B, C, H_out, W_out)
    """

    def __init__(self, in_channels: int = 3, scale_factor: float = 2.0,
                 align_corners: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h = int(H / self.scale_factor)
        target_w = int(W / self.scale_factor)

        b_bilinear = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear",
            align_corners=self.align_corners
        )
        b_nearest = F.interpolate(x, size=(target_h, target_w), mode="nearest")
        b_area = F.interpolate(x, size=(target_h, target_w), mode="area")

        fused = torch.cat([b_bilinear, b_nearest, b_area], dim=1)
        return self.fusion(fused) + b_area  # residual from area (best baseline)

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}, branches=[bilinear, nearest, area]"


@register_down("hwd_dual_path")
class HWDDualPathDown(nn.Module):
    """
    Haar Wavelet + MaxPool Dual-Path Downsampler (HWD-MP, IEEE 2024).

    Architecture:
        Input ──┬── Path A (Haar Wavelet Decomp → Conv1x1) ──┐
                │                                              ├── Add ── Output
                └── Path B (MaxPool)  ─────────────────────────┘

    Path A: Haar wavelet captures ALL frequency information losslessly,
            then a 1x1 conv compresses channels back to original count.
    Path B: MaxPool preserves strongest activations (edges, peaks).

    The combination ensures both complete spectral coverage (from HWD)
    and strong activation preservation (from MaxPool).

    Input:  (B, C, H, W)
    Output: (B, C, H//2, W//2)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Haar wavelet filters
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)
        self.register_buffer("haar_filters", filters)

        self.hwd_compress = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels * 2, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Path A: Haar wavelet decomposition
        x_flat = x.reshape(B * C, 1, H, W)
        haar_out = F.conv2d(x_flat, self.haar_filters, stride=2)
        haar_out = haar_out.reshape(B, C * 4, H // 2, W // 2)
        path_hwd = self.hwd_compress(haar_out)

        # Path B: MaxPool
        path_maxpool = self.maxpool(x)

        # Channel attention gate for adaptive fusion
        combined = torch.cat([path_hwd, path_maxpool], dim=1)
        B2, _, H2, W2 = path_hwd.shape
        gate_weight = self.gate(combined).view(B2, -1, 1, 1)

        return gate_weight * path_hwd + (1 - gate_weight) * path_maxpool

    def extra_repr(self) -> str:
        return "paths=[haar_wavelet, maxpool], fusion=channel_attention_gate"


@register_down("spd_conv")
class SPDConvDown(nn.Module):
    """
    Space-to-Depth + Non-Strided Convolution Downsampler (SPD-Conv).

    From "No More Strided Convolutions or Pooling" (2022).

    Instead of lossy strided conv or pooling, this:
    1. Rearranges spatial patches into channel dimension (zero info loss)
    2. Applies stride-1 convolution to fuse expanded channels

    The SPD step loses NO information. The subsequent conv learns to
    optimally compress the expanded representation.

    Input:  (B, C, H, W)
    Output: (B, out_channels, H//scale, W//scale)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = None, scale: int = 2):
        super().__init__()
        self.scale = scale
        out_channels = out_channels or in_channels
        expanded_channels = in_channels * scale * scale

        self.conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Space-to-Depth: (B, C, H, W) -> (B, C*scale^2, H//scale, W//scale)
        x = F.pixel_unshuffle(x, self.scale)
        return self.conv(x)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, method=space_to_depth+conv"


@register_down("conv_resize")
class ConvResizeDown(nn.Module):
    """
    Conv-Resize Block for Learned Fractional Downsampling.

    From "Convolutional Block Design for Learned Fractional Downsampling" (2022).

    Architecture: Conv(stride=1) → Differentiable Bilinear Resize

    The convolution learns to pre-filter the input optimally before
    the resize step, preserving important features while suppressing
    aliasing. Supports arbitrary (non-integer) scale factors.

    Input:  (B, C, H, W)
    Output: (B, C, H_out, W_out)
    """

    def __init__(self, in_channels: int = 3, scale_factor: float = 2.0,
                 kernel_size: int = 3, num_blocks: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

        blocks = []
        for _ in range(num_blocks):
            blocks.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
            ])
        self.pre_filter = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h = int(H / self.scale_factor)
        target_w = int(W / self.scale_factor)

        filtered = self.pre_filter(x)
        return F.interpolate(
            filtered, size=(target_h, target_w),
            mode="bilinear", align_corners=False
        )

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}, method=conv_then_resize"


@register_down("bilinear_conv_dual")
class BilinearConvDualDown(nn.Module):
    """
    Bilinear + Stacked Strided-Conv Dual-Branch with Attention Gate.

    Architecture:
        Input ──┬── Branch A (Bilinear resize) ────────────────┐
                │                                               ├── Attn Gate ── Weighted Sum ── Output
                └── Branch B (Conv stride=2, repeated N times) ─┘

    Branch A (Bilinear): Fixed smooth downsampling that preserves global
        spatial structure and low-frequency content. No learnable params.

    Branch B (Stacked Strided Convs): Progressive learnable downsampling.
        Each conv layer extracts and compresses features adaptively.
        Multiple layers capture multi-scale patterns at each reduction step.
        This branch learns WHAT to keep and what to discard.

    Attention Gate: Predicts per-channel, per-pixel weights to blend both
        branches. Edge/texture regions favor the conv branch (learned features),
        smooth regions favor bilinear (structure preservation).

    Why this works better:
        - Bilinear alone loses high-frequency detail (edges, textures)
        - Strided conv alone may discard global structure / introduce artifacts
        - Together: conv branch captures task-relevant features, bilinear
          branch ensures structural consistency. Attention selects optimally.

    Input:  (B, C, H, W)
    Output: (B, C, H // (2^num_stages), W // (2^num_stages))
    """

    def __init__(self, in_channels: int = 64, num_stages: int = 1,
                 kernel_size: int = 3, align_corners: bool = False):
        super().__init__()
        self.num_stages = num_stages
        self.total_scale = 2 ** num_stages
        self.align_corners = align_corners

        # Branch B: stacked strided convolutions (stride=2 each)
        conv_layers = []
        ch = in_channels
        for _ in range(num_stages):
            conv_layers.append(nn.Conv2d(ch, ch, kernel_size, stride=2,
                                         padding=kernel_size // 2, bias=False))
            conv_layers.append(nn.BatchNorm2d(ch))
            conv_layers.append(nn.SiLU(inplace=True))
        self.conv_branch = nn.Sequential(*conv_layers)

        # Attention gate: learns per-pixel blending weights
        self.attn_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h = H // self.total_scale
        target_w = W // self.total_scale

        # Branch A: bilinear (preserves global spatial structure)
        branch_bilinear = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear",
            align_corners=self.align_corners
        )

        # Branch B: stacked strided convolutions (learns what to keep)
        branch_conv = self.conv_branch(x)

        # Attention-gated fusion
        concat = torch.cat([branch_bilinear, branch_conv], dim=1)
        alpha = self.attn_gate(concat)

        return alpha * branch_bilinear + (1 - alpha) * branch_conv

    def extra_repr(self) -> str:
        return (f"num_stages={self.num_stages}, total_scale={self.total_scale}, "
                f"branches=[bilinear, stacked_strided_conv]")


@register_down("bilinear_conv_deep")
class BilinearConvDeepDualDown(nn.Module):
    """
    Deep Bilinear + Multi-Stage Strided-Conv with Progressive Attention Fusion.

    Like BilinearConvDualDown but applies attention fusion at EACH stage
    rather than only at the end. This gives finer control at every resolution level.

    Architecture (for num_stages=3, i.e., 8x total downsample):

        Stage 1 (H→H/2):
            Input ──┬── Bilinear(H/2) ──┐
                    │                    ├── Attn ── fused_1
                    └── Conv(s=2) ───────┘

        Stage 2 (H/2→H/4):
            fused_1 ──┬── Bilinear(H/4) ──┐
                      │                    ├── Attn ── fused_2
                      └── Conv(s=2) ───────┘

        Stage 3 (H/4→H/8):
            fused_2 ──┬── Bilinear(H/8) ──┐
                      │                    ├── Attn ── Output
                      └── Conv(s=2) ───────┘

    Each stage independently decides how much to trust bilinear vs learned conv
    at that particular resolution. Early stages may favor conv (high-freq features
    still present), later stages may favor bilinear (preserving remaining structure).

    Input:  (B, C, H, W)
    Output: (B, C, H // (2^num_stages), W // (2^num_stages))
    """

    def __init__(self, in_channels: int = 64, num_stages: int = 2,
                 kernel_size: int = 3, align_corners: bool = False):
        super().__init__()
        self.num_stages = num_stages
        self.total_scale = 2 ** num_stages
        self.align_corners = align_corners

        # Per-stage conv + attention
        self.conv_stages = nn.ModuleList()
        self.attn_stages = nn.ModuleList()

        for _ in range(num_stages):
            self.conv_stages.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride=2,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
            ))
            self.attn_stages.append(nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Sigmoid(),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for conv, attn in zip(self.conv_stages, self.attn_stages):
            H, W = out.shape[2], out.shape[3]
            target_h, target_w = H // 2, W // 2

            # Branch A: bilinear
            branch_bilinear = F.interpolate(
                out, size=(target_h, target_w), mode="bilinear",
                align_corners=self.align_corners
            )
            # Branch B: strided conv
            branch_conv = conv(out)

            # Attention-gated fusion at this stage
            concat = torch.cat([branch_bilinear, branch_conv], dim=1)
            alpha = attn(concat)
            out = alpha * branch_bilinear + (1 - alpha) * branch_conv

        return out

    def extra_repr(self) -> str:
        return (f"num_stages={self.num_stages}, total_scale={self.total_scale}, "
                f"fusion=per_stage_attention")


@register_up("bilinear_deconv_dual")
class BilinearDeconvDualUp(nn.Module):
    """
    Bilinear + Stacked TransposeConv Dual-Branch Upsampler with Attention Gate.

    Architecture:
        Input ──┬── Branch A (Bilinear upsample) ──────────────────────┐
                │                                                       ├── Attn Gate ── Weighted Sum ── Output
                └── Branch B (TransposeConv stride=2, repeated N times) ┘

    Branch A: Fixed bilinear gives smooth baseline structure.
    Branch B: Learnable transposed convolutions generate detail/sharpness.
    Attention gate blends them per-pixel (smooth areas → bilinear, detail → deconv).

    Input:  (B, C, H, W)
    Output: (B, C, H * (2^num_stages), W * (2^num_stages))
    """

    def __init__(self, in_channels: int = 64, num_stages: int = 1,
                 kernel_size: int = 4, align_corners: bool = False):
        super().__init__()
        self.num_stages = num_stages
        self.total_scale = 2 ** num_stages
        self.align_corners = align_corners

        # Branch B: stacked transposed convolutions
        deconv_layers = []
        ch = in_channels
        padding = (kernel_size - 2) // 2
        for _ in range(num_stages):
            deconv_layers.append(nn.ConvTranspose2d(ch, ch, kernel_size, stride=2,
                                                     padding=padding, bias=False))
            deconv_layers.append(nn.BatchNorm2d(ch))
            deconv_layers.append(nn.SiLU(inplace=True))
        self.deconv_branch = nn.Sequential(*deconv_layers)

        # Attention gate
        self.attn_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h = H * self.total_scale
        target_w = W * self.total_scale

        # Branch A: bilinear
        branch_bilinear = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear",
            align_corners=self.align_corners
        )

        # Branch B: transposed convolutions
        branch_deconv = self.deconv_branch(x)

        # Attention-gated fusion
        concat = torch.cat([branch_bilinear, branch_deconv], dim=1)
        alpha = self.attn_gate(concat)

        return alpha * branch_bilinear + (1 - alpha) * branch_deconv

    def extra_repr(self) -> str:
        return (f"num_stages={self.num_stages}, total_scale={self.total_scale}, "
                f"branches=[bilinear, stacked_transpose_conv]")


# =============================================================================
# Dual-Branch Fractional Upsamplers
# =============================================================================


@register_up("dual_branch")
class FractionalDualBranchUp(nn.Module):
    """
    Dual-Branch Fractional Upsampler with learnable feature fusion.

    Architecture:
        Input ──┬── Branch A (Bilinear) ──┐
                │                          ├── Concat ── Conv1x1 ── BN ── Output
                └── Branch B (Nearest)  ───┘

    Bilinear branch produces smooth upsampled features (good for gradients).
    Nearest branch preserves exact feature values (no blurring).
    Learnable fusion adaptively combines the best of both.

    Supports fractional scale factors.

    Input:  (B, C, H, W)
    Output: (B, C, H_out, W_out) where H_out = H * scale_factor
    """

    def __init__(self, in_channels: int = 3, scale_factor: float = 2.0,
                 align_corners: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h = int(H * self.scale_factor)
        target_w = int(W * self.scale_factor)

        branch_bilinear = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear",
            align_corners=self.align_corners
        )
        branch_nearest = F.interpolate(
            x, size=(target_h, target_w), mode="nearest"
        )

        fused = torch.cat([branch_bilinear, branch_nearest], dim=1)
        return self.fusion(fused)

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}, branches=[bilinear, nearest]"


@register_up("dual_branch_attn")
class DualBranchAttentionUp(nn.Module):
    """
    Dual-Branch Upsampler with spatial attention fusion.

    Uses a learned attention map to blend bilinear (smooth) and nearest
    (sharp) upsampled features per-pixel. Regions requiring sharpness
    (edges, textures) get more nearest-neighbor weight; smooth regions
    get more bilinear weight.

    Input:  (B, C, H, W)
    Output: (B, C, H_out, W_out)
    """

    def __init__(self, in_channels: int = 3, scale_factor: float = 2.0,
                 align_corners: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners

        self.attn_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h = int(H * self.scale_factor)
        target_w = int(W * self.scale_factor)

        branch_bilinear = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear",
            align_corners=self.align_corners
        )
        branch_nearest = F.interpolate(
            x, size=(target_h, target_w), mode="nearest"
        )

        concat = torch.cat([branch_bilinear, branch_nearest], dim=1)
        alpha = self.attn_gate(concat)

        return alpha * branch_bilinear + (1 - alpha) * branch_nearest

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}, fusion=attention_gate"


@register_up("dysample")
class DySampleUp(nn.Module):
    """
    DySample: Dynamic Upsampler via Point Sampling (ICCV 2023).

    Instead of fixed interpolation kernels, DySample learns to generate
    sampling offsets dynamically based on input content. It formulates
    upsampling as a point-sampling problem using grid_sample.

    Advantages over CARAFE/FADE:
      - No custom CUDA kernels needed
      - Fewer parameters and FLOPs
      - Lower latency
      - Outperforms across 5 dense prediction tasks

    Input:  (B, C, H, W)
    Output: (B, C, H * scale, W * scale)
    """

    def __init__(self, in_channels: int = 64, scale: int = 2, groups: int = 4,
                 style: str = "lp", use_dyscope: bool = False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ["lp", "pl"]
        if style == "pl":
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            offset_in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            offset_in_channels = in_channels
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(offset_in_channels, out_channels, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)

        self.use_dyscope = use_dyscope
        if use_dyscope:
            self.scope = nn.Conv2d(offset_in_channels, out_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid(h, h, indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def _sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(W, dtype=x.dtype, device=x.device) + 0.5
        coords = (
            torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(x.device)
        )
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W), coords,
            mode="bilinear", align_corners=False, padding_mode="border"
        ).view(B, -1, self.scale * H, self.scale * W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.style == "pl":
            x_ = F.pixel_shuffle(x, self.scale)
            if self.use_dyscope:
                offset = F.pixel_unshuffle(
                    self.offset(x_) * self.scope(x_).sigmoid(), self.scale
                ) * 0.5 + self.init_pos
            else:
                offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        else:
            if self.use_dyscope:
                offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
            else:
                offset = self.offset(x) * 0.25 + self.init_pos
        return self._sample(x, offset)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, style={self.style}, groups={self.groups}"


@register_up("carafe_lite")
class CARAFELiteUp(nn.Module):
    """
    Lightweight Content-Aware ReAssembly of Features (CARAFE-Lite).

    Simplified CARAFE: generates location-specific reassembly kernels from
    input content, then uses them to upsample features. Unlike fixed
    interpolation, each spatial location gets an adaptive kernel.

    Steps:
      1. Kernel prediction: compress channels → predict k*k kernels per location
      2. Content-aware reassembly: weighted sum over local neighborhood

    This version uses unfold + einsum for pure-PyTorch implementation
    (no custom CUDA ops needed).

    Input:  (B, C, H, W)
    Output: (B, C, H * scale, W * scale)
    """

    def __init__(self, in_channels: int = 64, scale: int = 2,
                 k_up: int = 5, compressed_channels: int = 16):
        super().__init__()
        self.scale = scale
        self.k_up = k_up
        self.pad = k_up // 2

        self.channel_compressor = nn.Conv2d(in_channels, compressed_channels, 1)
        self.kernel_predictor = nn.Sequential(
            nn.Conv2d(compressed_channels, compressed_channels, 3, padding=1, groups=compressed_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(compressed_channels, scale * scale * k_up * k_up, 1),
        )
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Predict reassembly kernels
        compressed = self.channel_compressor(x)
        kernels = self.kernel_predictor(compressed)  # (B, scale^2 * k^2, H, W)
        kernels = kernels.view(B, self.scale ** 2, self.k_up ** 2, H, W)
        kernels = F.softmax(kernels, dim=2)  # normalize kernel weights

        # Extract local neighborhoods via unfold
        x_padded = F.pad(x, [self.pad] * 4, mode="replicate")
        x_unfolded = x_padded.unfold(2, self.k_up, 1).unfold(3, self.k_up, 1)
        # x_unfolded: (B, C, H, W, k, k)
        x_unfolded = x_unfolded.reshape(B, C, H, W, self.k_up ** 2)

        # Apply kernels: for each sub-pixel position
        outputs = []
        for i in range(self.scale ** 2):
            kernel_i = kernels[:, i, :, :, :]  # (B, k^2, H, W)
            # Weighted sum: (B, C, H, W)
            out_i = torch.einsum("bchwk,bkhw->bchw", x_unfolded, kernel_i)
            outputs.append(out_i)

        # Arrange sub-pixels into spatial dimensions via pixel_shuffle
        stacked = torch.stack(outputs, dim=2)  # (B, C, scale^2, H, W)
        stacked = stacked.reshape(B, C * self.scale ** 2, H, W)
        return F.pixel_shuffle(stacked, self.scale)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, k_up={self.k_up}"


@register_up("resize_conv")
class ResizeConvUp(nn.Module):
    """
    Resize-Conv Upsampler (avoids checkerboard artifacts).

    Architecture: Bilinear Resize → Conv(stride=1)

    Unlike transposed convolution which is prone to checkerboard artifacts,
    this first resizes then refines with a convolution. The conv learns to
    sharpen and correct the bilinear upsampled output.

    Input:  (B, C, H, W)
    Output: (B, C, H * scale, W * scale)
    """

    def __init__(self, in_channels: int = 3, scale_factor: float = 2.0,
                 kernel_size: int = 3, num_blocks: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

        blocks = []
        for _ in range(num_blocks):
            blocks.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
            ])
        self.refine = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        target_h = int(H * self.scale_factor)
        target_w = int(W * self.scale_factor)

        upsampled = F.interpolate(
            x, size=(target_h, target_w),
            mode="bilinear", align_corners=False
        )
        return self.refine(upsampled)

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}, method=resize_then_conv"


# =============================================================================
# Advanced Techniques — Anti-Alias, Deformable, Gated, Frequency-Aware
# =============================================================================


@register_down("blur_pool")
class BlurPoolDown(nn.Module):
    """
    Anti-Aliased Downsampling (BlurPool) — ICML 2019.

    From "Making Convolutional Networks Shift-Invariant Again" (R. Zhang).

    Problem: Standard strided ops (maxpool, strided conv) violate the Nyquist
    sampling theorem, causing shift-variance and aliasing.

    Solution: Low-pass filter (blur) BEFORE subsampling.

    Architecture:
        Input → [Optional: Dense MaxPool/Conv] → Gaussian Blur → Subsample(stride)

    This makes the network shift-invariant AND improves accuracy.
    Compatible with any strided layer as a drop-in replacement.

    Input:  (B, C, H, W)
    Output: (B, C, H//stride, W//stride)
    """

    def __init__(self, in_channels: int = 64, stride: int = 2, filter_size: int = 4):
        super().__init__()
        self.stride = stride

        # Create Gaussian-like blur filter
        if filter_size == 1:
            filt = torch.tensor([1.0])
        elif filter_size == 2:
            filt = torch.tensor([1.0, 1.0])
        elif filter_size == 3:
            filt = torch.tensor([1.0, 2.0, 1.0])
        elif filter_size == 4:
            filt = torch.tensor([1.0, 3.0, 3.0, 1.0])
        elif filter_size == 5:
            filt = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        elif filter_size == 6:
            filt = torch.tensor([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif filter_size == 7:
            filt = torch.tensor([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        else:
            raise ValueError(f"filter_size must be 1-7, got {filter_size}")

        # Outer product for 2D filter
        filt = filt[:, None] * filt[None, :]
        filt = filt / filt.sum()

        # Depthwise filter: (C, 1, k, k)
        filt = filt.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.register_buffer("blur_filter", filt)
        self.pad = (filter_size - 1) // 2
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_padded = F.pad(x, [self.pad] * 4, mode="reflect")
        return F.conv2d(x_padded, self.blur_filter, stride=self.stride,
                        groups=self.in_channels)

    def extra_repr(self) -> str:
        k = self.blur_filter.shape[-1]
        return f"stride={self.stride}, blur_kernel={k}x{k}, anti_aliased=True"


@register_down("max_blur_pool")
class MaxBlurPoolDown(nn.Module):
    """
    MaxPool + BlurPool Anti-Aliased Downsampling.

    Replaces standard MaxPool2d. Decomposes max-pooling into:
      1. Dense max evaluation (stride=1)
      2. Anti-aliasing low-pass filter
      3. Subsampling

    This preserves the strong activation selection of MaxPool while
    eliminating the shift-variance caused by naive subsampling.

    Input:  (B, C, H, W)
    Output: (B, C, H//stride, W//stride)
    """

    def __init__(self, in_channels: int = 64, stride: int = 2,
                 kernel_size: int = 2, filter_size: int = 4):
        super().__init__()
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1,
                                    padding=kernel_size // 2)
        self.blur = BlurPoolDown(in_channels, stride=stride, filter_size=filter_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dense max (stride=1) then anti-aliased subsample
        x = self.maxpool(x)
        # Trim extra pixel from padding if kernel is even
        if self.maxpool.kernel_size % 2 == 0:
            x = x[:, :, :-1, :-1]
        return self.blur(x)

    def extra_repr(self) -> str:
        return f"stride={self.stride}, method=maxpool_dense+blur_subsample"


@register_down("gated_conv")
class GatedConvDown(nn.Module):
    """
    Gated Convolution Downsampler.

    Uses a learned gating mechanism to selectively pass features during
    downsampling. The gate learns which spatial regions contain important
    information that should survive the resolution reduction.

    Architecture:
        Input → Conv(stride=2) → Feature
        Input → Conv(stride=2) → Sigmoid → Gate
        Output = Feature * Gate

    The gate acts as a soft spatial attention mask that suppresses
    irrelevant regions and amplifies important features during downsampling.

    Input:  (B, C, H, W)
    Output: (B, C, H//scale, W//scale)
    """

    def __init__(self, in_channels: int = 64, scale: int = 2, kernel_size: int = 3):
        super().__init__()
        self.scale = scale
        padding = kernel_size // 2

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=scale,
                      padding=padding, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=scale,
                      padding=padding, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_conv(x)
        gate = self.gate_conv(x)
        return features * gate

    def extra_repr(self) -> str:
        return f"scale={self.scale}, method=gated_convolution"


@register_down("freq_split")
class FrequencySplitDown(nn.Module):
    """
    Frequency-Aware Split Downsampler.

    Explicitly decomposes input into low-frequency and high-frequency
    components, processes each with a specialized branch, then merges.

    Architecture:
        Input ──┬── AvgPool (low-freq) ──── Conv ──────────────────────────┐
                │                                                           ├── Weighted Add ── Output
                └── (Input - AvgPool) = high-freq ── Conv(stride=2) ───────┘

    Low-freq branch: smooth structural info (faces, object shapes)
    High-freq branch: edges, textures, fine details

    Each branch has its own learned processing, and a channel-attention
    gate decides the blending ratio per-channel.

    Input:  (B, C, H, W)
    Output: (B, C, H//2, W//2)
    """

    def __init__(self, in_channels: int = 64, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        # Low-frequency path
        self.low_freq_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        # High-frequency path
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=2,
                      padding=padding, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        # Channel attention for adaptive fusion
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels * 2, in_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels * 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decompose into frequency components
        low_freq = F.avg_pool2d(x, kernel_size=2, stride=1, padding=0)
        low_freq = F.pad(low_freq, [0, 1, 0, 1])  # restore size
        high_freq = x - low_freq

        # Process each frequency band
        low_out = self.low_freq_conv(self.low_freq_pool(x))
        high_out = self.high_freq_conv(high_freq)

        # Channel attention fusion
        combined = torch.cat([low_out, high_out], dim=1)
        B = combined.shape[0]
        attn = self.channel_attn(combined).view(B, 2, -1, 1, 1)
        alpha_low = attn[:, 0]
        alpha_high = attn[:, 1]

        return alpha_low * low_out + alpha_high * high_out

    def extra_repr(self) -> str:
        return "paths=[low_freq(avgpool), high_freq(residual)], fusion=channel_attention"


@register_down("deformable_down")
class DeformableConvDown(nn.Module):
    """
    Deformable Convolution Downsampler (DCNv2-style).

    Standard strided convolutions sample from fixed grid positions.
    Deformable convolution LEARNS where to sample from, adapting the
    receptive field to the content (e.g., following object boundaries).

    Architecture:
        Input → Offset Predictor (Conv) → offsets
        Input → Deformable Conv(stride=2, offsets) → Output

    The offset predictor generates 2D spatial offsets for each kernel
    position, allowing the convolution to "reach" for relevant features
    even during downsampling.

    Note: Uses a pure-PyTorch implementation via grid_sample (no custom CUDA).

    Input:  (B, C, H, W)
    Output: (B, C, H//scale, W//scale)
    """

    def __init__(self, in_channels: int = 64, scale: int = 2,
                 kernel_size: int = 3, offset_groups: int = 4):
        super().__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        self.offset_groups = offset_groups
        padding = kernel_size // 2

        # Offset prediction network
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, stride=scale, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 2, 2 * offset_groups * kernel_size * kernel_size, 1),
        )

        # Modulation mask (DCNv2)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, stride=scale, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 2, offset_groups * kernel_size * kernel_size, 1),
            nn.Sigmoid(),
        )

        # Main convolution weight
        self.weight = nn.Parameter(
            torch.empty(in_channels, in_channels // offset_groups, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

        self.stride = scale
        self.padding = padding
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offsets = self.offset_conv(x)
        mask = self.mask_conv(x)

        try:
            from torchvision.ops import deform_conv2d
            out = deform_conv2d(x, offsets, self.weight, mask=mask,
                                stride=self.stride, padding=self.padding)
        except ImportError:
            # Fallback: standard strided conv (loses deformable benefit)
            out = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding,
                           groups=self.offset_groups)

        return self.bn(out)

    def extra_repr(self) -> str:
        return (f"scale={self.scale}, kernel={self.kernel_size}, "
                f"offset_groups={self.offset_groups}, method=deformable_conv_v2")


@register_down("squeeze_excite_down")
class SqueezeExciteDown(nn.Module):
    """
    Squeeze-and-Excite Gated Downsample.

    Applies channel-wise squeeze-excitation attention AFTER downsampling
    to re-calibrate which channels survived the resolution reduction well.

    Architecture:
        Input → Conv(stride=2) → SE Block (recalibrate channels) → Output

    The SE block globally pools the downsampled features, learns
    inter-channel dependencies via FC layers, then re-weights channels.
    Channels that lost important info get suppressed; channels that
    preserved useful features get amplified.

    Input:  (B, C, H, W)
    Output: (B, C, H//scale, W//scale)
    """

    def __init__(self, in_channels: int = 64, scale: int = 2,
                 reduction: int = 4, kernel_size: int = 3):
        super().__init__()
        self.scale = scale
        padding = kernel_size // 2

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=scale,
                      padding=padding, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        # Squeeze-Excitation
        mid_channels = max(in_channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid_channels),
            nn.SiLU(inplace=True),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.downsample(x)
        B = out.shape[0]
        scale = self.se(out).view(B, -1, 1, 1)
        return out * scale

    def extra_repr(self) -> str:
        return f"scale={self.scale}, method=conv_stride+squeeze_excitation"


@register_down("progressive_dual")
class ProgressiveDualDown(nn.Module):
    """
    Progressive Dual-Branch with Residual Correction.

    Multi-stage downsampling where each stage:
      1. Bilinear branch provides stable baseline
      2. Conv branch learns residual correction
      3. Output = Bilinear + learned_residual

    This is more stable than attention-gated fusion because the bilinear
    path guarantees reasonable output even with untrained conv weights.
    The conv branch only needs to learn the DIFFERENCE (residual) from
    bilinear, which is easier to optimize.

    Input:  (B, C, H, W)
    Output: (B, C, H // (2^num_stages), W // (2^num_stages))
    """

    def __init__(self, in_channels: int = 64, num_stages: int = 2,
                 kernel_size: int = 3):
        super().__init__()
        self.num_stages = num_stages
        self.total_scale = 2 ** num_stages
        padding = kernel_size // 2

        self.residual_convs = nn.ModuleList()
        self.residual_scale = nn.ParameterList()

        for _ in range(num_stages):
            self.residual_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride=2,
                          padding=padding, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
            ))
            # Learnable residual scaling (starts at 0 = pure bilinear)
            self.residual_scale.append(nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for conv, scale in zip(self.residual_convs, self.residual_scale):
            H, W = out.shape[2], out.shape[3]
            # Bilinear baseline
            baseline = F.interpolate(out, size=(H // 2, W // 2),
                                     mode="bilinear", align_corners=False)
            # Learned residual correction (scaled from 0)
            residual = conv(out)
            out = baseline + scale.tanh() * residual
        return out

    def extra_repr(self) -> str:
        return (f"num_stages={self.num_stages}, total_scale={self.total_scale}, "
                f"method=bilinear_baseline+learned_residual")


@register_down("multi_kernel_down")
class MultiKernelDown(nn.Module):
    """
    Multi-Kernel Aggregation Downsampler.

    Uses multiple parallel convolutions with DIFFERENT kernel sizes
    to capture features at different spatial scales simultaneously.
    Small kernels capture fine details, large kernels capture context.

    Architecture:
        Input ──┬── Conv 3x3 (stride=2) ──┐
                ├── Conv 5x5 (stride=2) ──┼── Concat → Conv1x1 → Output
                ├── Conv 7x7 (stride=2) ──┤
                └── AvgPool (stride=2)  ───┘

    Input:  (B, C, H, W)
    Output: (B, C, H//2, W//2)
    """

    def __init__(self, in_channels: int = 64, kernels: list = None):
        super().__init__()
        kernels = kernels or [3, 5, 7]
        num_branches = len(kernels) + 1  # +1 for avgpool

        branch_channels = in_channels // num_branches
        remainder = in_channels - branch_channels * num_branches

        self.conv_branches = nn.ModuleList()
        for k in kernels:
            self.conv_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, k, stride=2,
                          padding=k // 2, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.SiLU(inplace=True),
            ))

        # AvgPool branch
        self.pool_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, branch_channels + remainder, 1, bias=False),
            nn.BatchNorm2d(branch_channels + remainder),
            nn.SiLU(inplace=True),
        )

        # Fusion
        total_out = branch_channels * len(kernels) + branch_channels + remainder
        self.fusion = nn.Sequential(
            nn.Conv2d(total_out, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [conv(x) for conv in self.conv_branches]
        branches.append(self.pool_branch(x))
        return self.fusion(torch.cat(branches, dim=1))

    def extra_repr(self) -> str:
        return "branches=[conv3x3, conv5x5, conv7x7, avgpool], fusion=concat+conv1x1"


@register_up("gated_deconv")
class GatedDeconvUp(nn.Module):
    """
    Gated Transposed Convolution Upsampler.

    Applies gating to transposed convolution output to suppress
    checkerboard artifacts and irrelevant upsampled features.

    Architecture:
        Input → TransposeConv → Features
        Input → TransposeConv → Sigmoid → Gate
        Output = Features * Gate

    Input:  (B, C, H, W)
    Output: (B, C, H * scale, W * scale)
    """

    def __init__(self, in_channels: int = 64, scale: int = 2, kernel_size: int = 4):
        super().__init__()
        self.scale = scale
        padding = (kernel_size - scale) // 2

        self.feature_deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size,
                               stride=scale, padding=padding, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.gate_deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size,
                               stride=scale, padding=padding, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_deconv(x)
        gate = self.gate_deconv(x)
        return features * gate

    def extra_repr(self) -> str:
        return f"scale={self.scale}, method=gated_transposed_conv"


@register_up("progressive_residual")
class ProgressiveResidualUp(nn.Module):
    """
    Progressive Residual Upsampler.

    Multi-stage upsampling where each stage:
      1. Bilinear provides smooth baseline
      2. SubPixel conv learns residual high-frequency detail
      3. Output = Bilinear + learned_detail

    The residual approach is stable: bilinear guarantees reasonable
    structure, conv only adds refinement. Residual scaling starts
    at 0 and grows during training.

    Input:  (B, C, H, W)
    Output: (B, C, H * (2^num_stages), W * (2^num_stages))
    """

    def __init__(self, in_channels: int = 64, num_stages: int = 2):
        super().__init__()
        self.num_stages = num_stages
        self.total_scale = 2 ** num_stages

        self.detail_convs = nn.ModuleList()
        self.residual_scale = nn.ParameterList()

        for _ in range(num_stages):
            self.detail_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels * 4),
                nn.SiLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
            ))
            self.residual_scale.append(nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for detail_conv, scale in zip(self.detail_convs, self.residual_scale):
            H, W = out.shape[2], out.shape[3]
            baseline = F.interpolate(out, size=(H * 2, W * 2),
                                     mode="bilinear", align_corners=False)
            detail = detail_conv(out)
            out = baseline + scale.tanh() * detail
        return out

    def extra_repr(self) -> str:
        return (f"num_stages={self.num_stages}, total_scale={self.total_scale}, "
                f"method=bilinear_baseline+subpixel_residual")


@register_up("freq_reconstruct")
class FrequencyReconstructUp(nn.Module):
    """
    Frequency-Aware Reconstruction Upsampler.

    Upsamples by separately generating low-frequency structure and
    high-frequency details, then combining.

    Architecture:
        Input ──┬── Bilinear (2x) → produces low-freq base ──────────────┐
                │                                                          ├── Add ── Output
                └── PixelShuffle branch → produces high-freq detail ──────┘

    The bilinear branch guarantees smooth global structure.
    The PixelShuffle branch specializes in generating missing high-freq
    details (edges, textures) that bilinear cannot produce.

    Input:  (B, C, H, W)
    Output: (B, C, H * 2, W * 2)
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()

        # High-frequency detail generator
        self.detail_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 4),
            nn.SiLU(inplace=True),
            nn.PixelShuffle(2),
        )

        # Learnable blend weight (starts favoring bilinear, learns to add detail)
        self.blend = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        low_freq = F.interpolate(x, size=(H * 2, W * 2),
                                  mode="bilinear", align_corners=False)
        high_freq = self.detail_gen(x)
        return low_freq + self.blend * high_freq

    def extra_repr(self) -> str:
        return "paths=[bilinear_base, pixelshuffle_detail], fusion=weighted_add"


@register_up("multi_kernel_up")
class MultiKernelUp(nn.Module):
    """
    Multi-Kernel Aggregation Upsampler.

    Multiple parallel transposed convolutions with different kernel sizes
    capture upsampled features at multiple scales. Small kernels produce
    sharp local detail; large kernels produce smooth global context.

    Architecture:
        Input ──┬── DeConv 2x2 (stride=2) ──┐
                ├── DeConv 4x4 (stride=2) ──┼── Concat → Conv1x1 → Output
                ├── DeConv 6x6 (stride=2) ──┤
                └── Bilinear (2x)  ──────────┘

    Input:  (B, C, H, W)
    Output: (B, C, H * 2, W * 2)
    """

    def __init__(self, in_channels: int = 64, kernels: list = None):
        super().__init__()
        kernels = kernels or [2, 4, 6]
        num_branches = len(kernels) + 1

        branch_channels = in_channels // num_branches
        remainder = in_channels - branch_channels * num_branches

        self.deconv_branches = nn.ModuleList()
        for k in kernels:
            padding = (k - 2) // 2
            self.deconv_branches.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, branch_channels, k,
                                   stride=2, padding=padding, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.SiLU(inplace=True),
            ))

        # Bilinear branch with channel reduction
        self.bilinear_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels + remainder, 1, bias=False),
            nn.BatchNorm2d(branch_channels + remainder),
            nn.SiLU(inplace=True),
        )

        total_out = branch_channels * len(kernels) + branch_channels + remainder
        self.fusion = nn.Sequential(
            nn.Conv2d(total_out, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        branches = [deconv(x) for deconv in self.deconv_branches]

        bilinear_up = F.interpolate(x, size=(H * 2, W * 2),
                                     mode="bilinear", align_corners=False)
        branches.append(self.bilinear_branch(bilinear_up))

        return self.fusion(torch.cat(branches, dim=1))

    def extra_repr(self) -> str:
        return "branches=[deconv2x2, deconv4x4, deconv6x6, bilinear], fusion=concat+conv1x1"
