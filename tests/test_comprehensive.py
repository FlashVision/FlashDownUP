"""Comprehensive tests for FlashDownUP: all operators, pipelines, metrics, CLI."""

from __future__ import annotations

import torch
import pytest


# ---------------------------------------------------------------------------
# Lossless: SpaceToDepth / DepthToSpace
# ---------------------------------------------------------------------------


class TestSpaceToDepth:
    def test_output_shape_scale2(self):
        from flashdownup import SpaceToDepth

        op = SpaceToDepth(scale=2)
        x = torch.randn(2, 3, 64, 64)
        y = op(x)
        assert y.shape == (2, 12, 32, 32)

    def test_output_shape_scale4(self):
        from flashdownup import SpaceToDepth

        op = SpaceToDepth(scale=4)
        x = torch.randn(1, 3, 64, 64)
        y = op(x)
        assert y.shape == (1, 48, 16, 16)

    def test_roundtrip(self):
        from flashdownup import DepthToSpace, SpaceToDepth

        down = SpaceToDepth(scale=2)
        up = DepthToSpace(scale=2)
        x = torch.randn(1, 3, 64, 64)
        assert torch.allclose(up(down(x)), x, atol=1e-6)

    def test_gradient_flow(self):
        from flashdownup import SpaceToDepth

        op = SpaceToDepth(scale=2)
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        y = op(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestDepthToSpace:
    def test_output_shape(self):
        from flashdownup import DepthToSpace

        op = DepthToSpace(scale=2)
        x = torch.randn(2, 12, 32, 32)
        y = op(x)
        assert y.shape == (2, 3, 64, 64)

    def test_scale3(self):
        from flashdownup import DepthToSpace

        op = DepthToSpace(scale=3)
        x = torch.randn(1, 27, 10, 10)
        y = op(x)
        assert y.shape == (1, 3, 30, 30)


# ---------------------------------------------------------------------------
# Lossless: DWT Haar
# ---------------------------------------------------------------------------


class TestDWTHaar:
    def test_output_shape(self):
        from flashdownup import DWTHaarDown

        op = DWTHaarDown()
        x = torch.randn(1, 3, 64, 64)
        y = op(x)
        assert y.shape == (1, 12, 32, 32)

    def test_roundtrip(self):
        from flashdownup import DWTHaarDown, IDWTHaarUp

        down = DWTHaarDown()
        up = IDWTHaarUp()
        x = torch.randn(1, 3, 64, 64)
        reconstructed = up(down(x))
        assert torch.allclose(x, reconstructed, atol=1e-5)

    def test_multichannel(self):
        from flashdownup import DWTHaarDown, IDWTHaarUp

        down = DWTHaarDown()
        up = IDWTHaarUp()
        x = torch.randn(2, 16, 32, 32)
        encoded = down(x)
        assert encoded.shape == (2, 64, 16, 16)
        assert torch.allclose(up(encoded), x, atol=1e-5)

    def test_energy_preservation(self):
        from flashdownup import DWTHaarDown

        op = DWTHaarDown()
        x = torch.randn(1, 3, 64, 64)
        y = op(x)
        energy_in = (x**2).sum().item()
        energy_out = (y**2).sum().item()
        assert abs(energy_in - energy_out) / energy_in < 0.01


# ---------------------------------------------------------------------------
# Lossy Down
# ---------------------------------------------------------------------------


class TestFlashDown:
    @pytest.mark.parametrize("method", ["bilinear", "bicubic", "nearest", "maxpool", "avgpool"])
    def test_interpolation_shapes(self, method):
        from flashdownup import FlashDown

        down = FlashDown(method, scale=2)
        x = torch.randn(1, 3, 64, 64)
        y = down(x)
        assert y.shape == (1, 3, 32, 32)

    def test_strided_conv(self):
        from flashdownup import FlashDown

        down = FlashDown("strided_conv", in_channels=3, scale=2)
        x = torch.randn(1, 3, 64, 64)
        y = down(x)
        assert y.shape == (1, 3, 32, 32)

    def test_s2d_method(self):
        from flashdownup import FlashDown

        down = FlashDown("s2d", scale=2)
        x = torch.randn(1, 3, 64, 64)
        y = down(x)
        assert y.shape == (1, 12, 32, 32)

    def test_dwt_haar_method(self):
        from flashdownup import FlashDown

        down = FlashDown("dwt_haar")
        x = torch.randn(1, 3, 64, 64)
        y = down(x)
        assert y.shape == (1, 12, 32, 32)

    def test_available_methods(self):
        from flashdownup import FlashDown

        methods = FlashDown.available_methods()
        assert "s2d" in methods
        assert "dwt_haar" in methods
        assert "bilinear" in methods

    def test_invalid_method_raises(self):
        from flashdownup import FlashDown

        with pytest.raises(ValueError):
            FlashDown("nonexistent_xyz")

    @pytest.mark.parametrize("scale", [2, 4])
    def test_scale_factors(self, scale):
        from flashdownup import FlashDown

        down = FlashDown("bilinear", scale=scale)
        x = torch.randn(1, 3, 64, 64)
        y = down(x)
        assert y.shape == (1, 3, 64 // scale, 64 // scale)


# ---------------------------------------------------------------------------
# Lossy Up
# ---------------------------------------------------------------------------


class TestFlashUp:
    @pytest.mark.parametrize("method", ["bilinear", "bicubic", "nearest"])
    def test_interpolation_shapes(self, method):
        from flashdownup import FlashUp

        up = FlashUp(method, scale=2)
        x = torch.randn(1, 3, 32, 32)
        y = up(x)
        assert y.shape == (1, 3, 64, 64)

    def test_transpose_conv(self):
        from flashdownup import FlashUp

        up = FlashUp("transpose_conv", in_channels=3, scale=2)
        x = torch.randn(1, 3, 32, 32)
        y = up(x)
        assert y.shape == (1, 3, 64, 64)

    def test_subpixel(self):
        from flashdownup import FlashUp

        up = FlashUp("subpixel", in_channels=3, scale=2)
        x = torch.randn(1, 3, 32, 32)
        y = up(x)
        assert y.shape == (1, 3, 64, 64)

    def test_d2s_method(self):
        from flashdownup import FlashUp

        up = FlashUp("d2s", scale=2)
        x = torch.randn(1, 12, 32, 32)
        y = up(x)
        assert y.shape == (1, 3, 64, 64)

    def test_idwt_haar_method(self):
        from flashdownup import FlashUp

        up = FlashUp("idwt_haar")
        x = torch.randn(1, 12, 32, 32)
        y = up(x)
        assert y.shape == (1, 3, 64, 64)

    def test_available_methods(self):
        from flashdownup import FlashUp

        methods = FlashUp.available_methods()
        assert "d2s" in methods
        assert "idwt_haar" in methods
        assert "subpixel" in methods

    def test_invalid_method_raises(self):
        from flashdownup import FlashUp

        with pytest.raises(ValueError):
            FlashUp("nonexistent_xyz")


# ---------------------------------------------------------------------------
# Image Quality Metrics (PSNR, SSIM approximation)
# ---------------------------------------------------------------------------


class TestImageQualityMetrics:
    def test_psnr_identical(self):
        x = torch.randn(1, 3, 32, 32)
        mse = ((x - x) ** 2).mean().item()
        assert mse == 0.0

    def test_psnr_noisy(self):
        x = torch.randn(1, 3, 32, 32)
        noise = torch.randn_like(x) * 0.1
        y = x + noise
        mse = ((x - y) ** 2).mean().item()
        psnr = 10 * torch.log10(torch.tensor(1.0) / torch.tensor(mse)).item()
        assert psnr > 0
        assert psnr < 100

    def test_ssim_identical(self):
        x = torch.rand(1, 1, 16, 16)
        mu_x = x.mean()
        var_x = x.var()
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2
        ssim = ((2 * mu_x * mu_x + c1) * (2 * var_x + c2)) / ((mu_x**2 + mu_x**2 + c1) * (var_x + var_x + c2))
        assert ssim.item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Analytics: profile_operator
# ---------------------------------------------------------------------------


class TestAnalytics:
    def test_profile_operator(self):
        from flashdownup import FlashDown
        from flashdownup.analytics import profile_operator

        op = FlashDown("bilinear", scale=2)
        result = profile_operator(op, input_shape=(1, 3, 64, 64), warmup=2, iterations=5)
        assert result["mean_ms"] > 0
        assert result["output_shape"] == [1, 3, 32, 32]
        assert result["params"] == 0

    def test_compare_methods(self):
        from flashdownup.analytics import compare_methods

        results = compare_methods(["bilinear", "nearest"], direction="down", input_shape=(1, 3, 64, 64), scale=2)
        assert len(results) == 2
        assert all(r["method"] in ("bilinear", "nearest") for r in results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestFlashDownUPCLI:
    def test_list_command(self, capsys):
        import argparse

        from flashdownup.cli import _cmd_list

        _cmd_list(argparse.Namespace(type="all"))
        captured = capsys.readouterr()
        assert "Downsamplers" in captured.out
        assert "Upsamplers" in captured.out

    def test_list_down_only(self, capsys):
        import argparse

        from flashdownup.cli import _cmd_list

        _cmd_list(argparse.Namespace(type="down"))
        captured = capsys.readouterr()
        assert "Downsamplers" in captured.out
        assert "Upsamplers" not in captured.out


# ---------------------------------------------------------------------------
# Integration: low-res → upscale → evaluate quality
# ---------------------------------------------------------------------------


class TestIntegrationDownUP:
    def test_downscale_then_upscale(self):
        from flashdownup import FlashDown, FlashUp

        x = torch.randn(1, 3, 64, 64)
        down = FlashDown("bilinear", scale=2)
        up = FlashUp("bilinear", scale=2)

        low_res = down(x)
        assert low_res.shape == (1, 3, 32, 32)

        upscaled = up(low_res)
        assert upscaled.shape == (1, 3, 64, 64)

        mse = ((x - upscaled) ** 2).mean().item()
        assert mse > 0

    def test_lossless_roundtrip_pipeline(self):
        from flashdownup import FlashDown, FlashUp

        x = torch.randn(1, 3, 64, 64)
        down = FlashDown("s2d", scale=2)
        up = FlashUp("d2s", scale=2)

        encoded = down(x)
        reconstructed = up(encoded)
        assert torch.allclose(x, reconstructed, atol=1e-6)

    def test_super_resolution_quality(self):
        from flashdownup import FlashUp

        low_res = torch.rand(1, 3, 16, 16)
        up = FlashUp("bicubic", scale=4)
        high_res = up(low_res)
        assert high_res.shape == (1, 3, 64, 64)
        assert high_res.min() >= -0.5
        assert high_res.max() <= 1.5
