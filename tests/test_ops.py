"""Tests for FlashDownUP operators."""

import torch
import pytest


class TestSpaceToDepth:
    def test_output_shape(self):
        from flashdownup import SpaceToDepth
        op = SpaceToDepth(scale=2)
        x = torch.randn(2, 3, 64, 64)
        y = op(x)
        assert y.shape == (2, 12, 32, 32)

    def test_scale_4(self):
        from flashdownup import SpaceToDepth
        op = SpaceToDepth(scale=4)
        x = torch.randn(1, 3, 64, 64)
        y = op(x)
        assert y.shape == (1, 48, 16, 16)

    def test_lossless_roundtrip(self):
        from flashdownup import SpaceToDepth, DepthToSpace
        down = SpaceToDepth(scale=2)
        up = DepthToSpace(scale=2)
        x = torch.randn(1, 3, 64, 64)
        reconstructed = up(down(x))
        assert torch.allclose(x, reconstructed, atol=1e-6)


class TestDepthToSpace:
    def test_output_shape(self):
        from flashdownup import DepthToSpace
        op = DepthToSpace(scale=2)
        x = torch.randn(2, 12, 32, 32)
        y = op(x)
        assert y.shape == (2, 3, 64, 64)


class TestDWTHaar:
    def test_output_shape(self):
        from flashdownup import DWTHaarDown
        op = DWTHaarDown()
        x = torch.randn(1, 3, 64, 64)
        y = op(x)
        assert y.shape == (1, 12, 32, 32)

    def test_lossless_roundtrip(self):
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
        reconstructed = up(encoded)
        assert torch.allclose(x, reconstructed, atol=1e-5)


class TestLossyDown:
    @pytest.mark.parametrize("method", ["bilinear", "bicubic", "nearest", "maxpool", "avgpool"])
    def test_output_shape(self, method):
        from flashdownup import FlashDown
        down = FlashDown(method, scale=2)
        x = torch.randn(1, 3, 64, 64)
        y = down(x)
        assert y.shape == (1, 3, 32, 32)

    def test_strided_conv_shape(self):
        from flashdownup import FlashDown
        down = FlashDown("strided_conv", in_channels=3, scale=2)
        x = torch.randn(1, 3, 64, 64)
        y = down(x)
        assert y.shape == (1, 3, 32, 32)


class TestLossyUp:
    @pytest.mark.parametrize("method", ["bilinear", "bicubic", "nearest"])
    def test_output_shape(self, method):
        from flashdownup import FlashUp
        up = FlashUp(method, scale=2)
        x = torch.randn(1, 3, 32, 32)
        y = up(x)
        assert y.shape == (1, 3, 64, 64)

    def test_transpose_conv_shape(self):
        from flashdownup import FlashUp
        up = FlashUp("transpose_conv", in_channels=3, scale=2)
        x = torch.randn(1, 3, 32, 32)
        y = up(x)
        assert y.shape == (1, 3, 64, 64)

    def test_subpixel_shape(self):
        from flashdownup import FlashUp
        up = FlashUp("subpixel", in_channels=3, scale=2)
        x = torch.randn(1, 3, 32, 32)
        y = up(x)
        assert y.shape == (1, 3, 64, 64)


class TestFlashDownUp:
    def test_list_methods(self):
        from flashdownup import FlashDown, FlashUp
        downs = FlashDown.available_methods()
        ups = FlashUp.available_methods()
        assert "s2d" in downs
        assert "dwt_haar" in downs
        assert "bilinear" in downs
        assert "d2s" in ups
        assert "idwt_haar" in ups
        assert "subpixel" in ups

    def test_invalid_method(self):
        from flashdownup import FlashDown
        with pytest.raises(ValueError):
            FlashDown("nonexistent_method")
