from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Hashable

import torch

from .zarr_store import UPDZarrStore
from .cache import LRUCache


def _norm_keys(keys: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if keys is None:
        return tuple()
    return tuple(sorted([str(k) for k in keys]))


@dataclass
class ZarrCacheConfig:
    max_samples: int = 256       # cache of full samples
    max_fields: int = 2048       # cache of individual field reads
    enable_field_cache: bool = True


class CachedUPDZarrStore:
    """
    Cached wrapper around UPDZarrStore.
    """
    def __init__(self, root: str, *, cache: Optional[ZarrCacheConfig] = None, mode: str = "r"):
        self.store = UPDZarrStore(root, mode=mode)
        self.cache_cfg = cache or ZarrCacheConfig()

        self.sample_cache = LRUCache(max_items=self.cache_cfg.max_samples)
        self.field_cache = LRUCache(max_items=self.cache_cfg.max_fields)

    def count(self) -> int:
        return self.store.count()

    def manifest(self) -> Dict[str, Any]:
        return self.store.manifest()

    def _sample_key(
        self,
        i: int,
        fields: Optional[Sequence[str]],
        coords: Optional[Sequence[str]],
        device: str | torch.device,
        dtype: Optional[torch.dtype],
    ) -> Hashable:
        dev = str(device)
        dt = str(dtype) if dtype is not None else ""
        return ("sample", int(i), _norm_keys(fields), _norm_keys(coords), dev, dt)

    def _field_key(
        self,
        i: int,
        kind: str,      # "field" or "coord"
        name: str,
        device: str | torch.device,
        dtype: Optional[torch.dtype],
    ) -> Hashable:
        dev = str(device)
        dt = str(dtype) if dtype is not None else ""
        return (kind, int(i), str(name), dev, dt)

    def read_sample(
        self,
        i: int,
        *,
        fields: Optional[Sequence[str]] = None,
        coords: Optional[Sequence[str]] = None,
        device: str | torch.device = "cpu",
        dtype: Optional[torch.dtype] = None,
        sample_ctor=None,
        use_sample_cache: bool = True,
    ):
        # 1) full sample cache
        sk = self._sample_key(i, fields, coords, device, dtype)
        if use_sample_cache:
            cached = self.sample_cache.get(sk)
            if cached is not None:
                return cached

        # 2) field-level caching (optional)
        if self.cache_cfg.enable_field_cache:
            fk = self.store.field_keys()
            ck = self.store.coord_keys()
            req_fields = list(fields) if fields is not None else fk
            req_coords = list(coords) if coords is not None else ck

            out_fields: Dict[str, torch.Tensor] = {}
            out_coords: Dict[str, torch.Tensor] = {}

            for name in req_fields:
                k = self._field_key(i, "field", name, device, dtype)
                v = self.field_cache.get(k)
                if v is None:
                    # read just this field from zarr (one chunk)
                    # Use underlying store read_sample but requesting only one key to minimize work
                    s1 = self.store.read_sample(
                        i, fields=[name], coords=[], device=device, dtype=dtype, sample_ctor=lambda f, c, m: (f, m)
                    )
                    fdict, meta = s1
                    if name in fdict:
                        v = fdict[name]
                        self.field_cache.put(k, v)
                if v is not None:
                    out_fields[name] = v

            for name in req_coords:
                k = self._field_key(i, "coord", name, device, dtype)
                v = self.field_cache.get(k)
                if v is None:
                    s1 = self.store.read_sample(
                        i, fields=[], coords=[name], device=device, dtype=dtype, sample_ctor=lambda f, c, m: (c, m)
                    )
                    cdict, meta = s1
                    if name in cdict:
                        v = cdict[name]
                        self.field_cache.put(k, v)
                if v is not None:
                    out_coords[name] = v

            meta = self.store.meta(i)

            if sample_ctor is None:
                try:
                    from .physical_sample import PhysicalSample
                    sample = PhysicalSample(fields=out_fields, coords=out_coords, meta=meta)
                except Exception:
                    sample = {"fields": out_fields, "coords": out_coords, "meta": meta}
            else:
                sample = sample_ctor(out_fields, out_coords, meta)

            # put in sample cache for this exact request signature
            if use_sample_cache:
                self.sample_cache.put(sk, sample)
            return sample

        # 3) fallback: no field cache
        sample = self.store.read_sample(i, fields=fields, coords=coords, device=device, dtype=dtype, sample_ctor=sample_ctor)
        if use_sample_cache:
            self.sample_cache.put(sk, sample)
        return sample

    def cache_stats(self) -> Dict[str, Any]:
        return {
            "sample_cache": {
                "size": len(self.sample_cache),
                "hits": self.sample_cache.stats.hits,
                "misses": self.sample_cache.stats.misses,
                "evictions": self.sample_cache.stats.evictions,
            },
            "field_cache": {
                "size": len(self.field_cache),
                "hits": self.field_cache.stats.hits,
                "misses": self.field_cache.stats.misses,
                "evictions": self.field_cache.stats.evictions,
            },
        }
