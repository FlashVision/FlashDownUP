"""Downsampling and upsampling operator implementations."""

# Import all operator modules to trigger registry registration
from flashdownup.ops import lossless  # noqa: F401
from flashdownup.ops import lossy  # noqa: F401
from flashdownup.ops import fractional  # noqa: F401
