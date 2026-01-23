from __future__ import annotations

from typing import Any, Dict, Optional
import torch


def _pin(x: Any) -> Any:
    if torch.is_tensor(x) and x.device.type == "cpu":
        try:
            return x.pin_memory()
        except Exception:
            return x
    return x


def _to(x: Any, device: torch.device, dtype: Optional[torch.dtype], non_blocking: bool) -> Any:
    if torch.is_tensor(x):
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype=dtype)
        return x.to(device=device, non_blocking=non_blocking)
    return x


def pin_sample(sample: Any) -> Any:
    """
    Pins all tensor fields/coords in a PhysicalSample-like object.

    Supports:
      - sample.fields / sample.coords
      - dict {"fields":..., "coords":...}
    """
    if hasattr(sample, "fields") and hasattr(sample, "coords"):
        sample.fields = {k: _pin(v) for k, v in sample.fields.items()}
        sample.coords = {k: _pin(v) for k, v in sample.coords.items()}
        return sample

    if isinstance(sample, dict) and "fields" in sample and "coords" in sample:
        sample["fields"] = {k: _pin(v) for k, v in sample["fields"].items()}
        sample["coords"] = {k: _pin(v) for k, v in sample["coords"].items()}
        return sample

    return sample


def to_device_sample(sample: Any, device: str | torch.device, *, dtype: Optional[torch.dtype] = None, non_blocking: bool = True) -> Any:
    """
    Moves all tensor fields/coords to target device.

    Supports:
      - sample.fields / sample.coords
      - dict {"fields":..., "coords":...}
    """
    dev = torch.device(device)

    if hasattr(sample, "fields") and hasattr(sample, "coords"):
        sample.fields = {k: _to(v, dev, dtype, non_blocking) for k, v in sample.fields.items()}
        sample.coords = {k: _to(v, dev, dtype, non_blocking) for k, v in sample.coords.items()}
        return sample

    if isinstance(sample, dict) and "fields" in sample and "coords" in sample:
        sample["fields"] = {k: _to(v, dev, dtype, non_blocking) for k, v in sample["fields"].items()}
        sample["coords"] = {k: _to(v, dev, dtype, non_blocking) for k, v in sample["coords"].items()}
        return sample

    return sample
