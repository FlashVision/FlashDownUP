"""Registry for downsampling and upsampling operators."""

from typing import Dict, Type, Any

import torch.nn as nn


_DOWN_REGISTRY: Dict[str, Type[nn.Module]] = {}
_UP_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_down(name: str):
    """Decorator to register a downsampling operator."""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _DOWN_REGISTRY:
            raise ValueError(f"Downsampler '{name}' already registered.")
        _DOWN_REGISTRY[name] = cls
        return cls
    return decorator


def register_up(name: str):
    """Decorator to register an upsampling operator."""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _UP_REGISTRY:
            raise ValueError(f"Upsampler '{name}' already registered.")
        _UP_REGISTRY[name] = cls
        return cls
    return decorator


def get_downsampler(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered downsampler by name."""
    if name not in _DOWN_REGISTRY:
        available = ", ".join(sorted(_DOWN_REGISTRY.keys()))
        raise ValueError(f"Unknown downsampler '{name}'. Available: {available}")
    return _DOWN_REGISTRY[name](**kwargs)


def get_upsampler(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered upsampler by name."""
    if name not in _UP_REGISTRY:
        available = ", ".join(sorted(_UP_REGISTRY.keys()))
        raise ValueError(f"Unknown upsampler '{name}'. Available: {available}")
    return _UP_REGISTRY[name](**kwargs)


def list_downsamplers() -> list:
    """List all registered downsampler names."""
    return sorted(_DOWN_REGISTRY.keys())


def list_upsamplers() -> list:
    """List all registered upsampler names."""
    return sorted(_UP_REGISTRY.keys())
