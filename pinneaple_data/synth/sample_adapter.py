from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable, Union

import torch


@runtime_checkable
class PhysicalSampleLike(Protocol):
    fields: Dict[str, Any]
    coords: Dict[str, Any]
    meta: Dict[str, Any]


def has_pinnego_physical_sample() -> bool:
    try:
        from pinneaple_data.physical_sample import PhysicalSample  # noqa: F401
        return True
    except Exception:
        return False


def _torchify_tree(obj: Any, *, device=None, dtype=None) -> Any:
    """
    Convert numpy/lists -> torch.Tensor recursively when possible.
    Keeps non-tensor metadata as is.
    """
    if isinstance(obj, torch.Tensor):
        t = obj
        if device is not None:
            t = t.to(device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t

    # try tensor conversion for arrays/lists of numeric
    if isinstance(obj, (list, tuple)):
        # keep lists of strings intact
        if len(obj) > 0 and all(isinstance(x, str) for x in obj):
            return obj
        try:
            return torch.tensor(obj, device=device, dtype=dtype)  # type: ignore[arg-type]
        except Exception:
            return [_torchify_tree(x, device=device, dtype=dtype) for x in obj]

    if isinstance(obj, dict):
        return {k: _torchify_tree(v, device=device, dtype=dtype) for k, v in obj.items()}

    # numpy arrays
    try:
        import numpy as np  # noqa
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj).to(device=device, dtype=dtype)  # type: ignore[arg-type]
    except Exception:
        pass

    return obj


def to_physical_sample(
    sample_like: Any,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Any:
    """
    Best-effort conversion:
      - if already a PhysicalSample -> return
      - if duck-typed {fields, coords, meta} -> convert
      - else if dict with keys -> interpret

    Returns:
      - pinneaple_data.physical_sample.PhysicalSample if available
      - else returns a minimal dataclass fallback (SynthPhysicalSample)
    """
    device = torch.device(device) if device is not None and not isinstance(device, torch.device) else device

    # Already a PhysicalSample?
    if has_pinnego_physical_sample():
        from pinneaple_data.physical_sample import PhysicalSample  # type: ignore
        if isinstance(sample_like, PhysicalSample):
            return sample_like

    # Extract fields/coords/meta
    fields = None
    coords = None
    meta = None

    if hasattr(sample_like, "fields") and hasattr(sample_like, "coords") and hasattr(sample_like, "meta"):
        fields = getattr(sample_like, "fields")
        coords = getattr(sample_like, "coords")
        meta = getattr(sample_like, "meta")
    elif isinstance(sample_like, dict):
        # common patterns
        if "fields" in sample_like:
            fields = sample_like.get("fields")
            coords = sample_like.get("coords", {})
            meta = sample_like.get("meta", {})
        elif "state" in sample_like:
            fields = sample_like.get("state")
            coords = sample_like.get("coords", {})
            meta = sample_like.get("meta", {})
        else:
            # interpret dict as fields directly
            fields = {k: v for k, v in sample_like.items() if k not in ("coords", "meta")}
            coords = sample_like.get("coords", {})
            meta = sample_like.get("meta", {})
    else:
        raise TypeError("Cannot convert object to PhysicalSample: expected {fields,coords,meta} or dict-like.")

    fields_t = _torchify_tree(fields, device=device, dtype=dtype)
    coords_t = _torchify_tree(coords, device=device, dtype=dtype)
    meta = dict(meta) if isinstance(meta, dict) else {"meta": meta}

    # Preferred: real PhysicalSample
    if has_pinnego_physical_sample():
        from pinneaple_data.physical_sample import PhysicalSample  # type: ignore
        # Many projects define PhysicalSample differently; use the most conservative constructor:
        try:
            return PhysicalSample(fields=fields_t, coords=coords_t, meta=meta)
        except TypeError:
            # fallback constructor names
            try:
                return PhysicalSample(state=fields_t, coords=coords_t, meta=meta)
            except TypeError:
                # last resort: create and set attrs
                ps = PhysicalSample()
                setattr(ps, "fields", fields_t)
                setattr(ps, "coords", coords_t)
                setattr(ps, "meta", meta)
                return ps

    # Fallback minimal
    @dataclass
    class SynthPhysicalSample:
        fields: Dict[str, Any]
        coords: Dict[str, Any]
        meta: Dict[str, Any]

    return SynthPhysicalSample(fields=fields_t, coords=coords_t, meta=meta)
