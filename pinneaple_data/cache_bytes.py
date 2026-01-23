from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional, Tuple
from collections import OrderedDict
import threading

import torch


@dataclass
class ByteCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_in_use: int = 0


def default_weigher(value: Any) -> int:
    """
    Estimate memory usage in bytes.

    - torch.Tensor: nbytes
    - dict with tensors: sum
    - PhysicalSample-like: sum(fields)+sum(coords)
    - fallback: 1 KB
    """
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())

    # PhysicalSample-like
    if hasattr(value, "fields") and hasattr(value, "coords"):
        total = 0
        for v in value.fields.values():
            total += default_weigher(v)
        for v in value.coords.values():
            total += default_weigher(v)
        return total

    if isinstance(value, dict):
        total = 0
        for v in value.values():
            total += default_weigher(v)
        return total

    return 1024


class ByteLRUCache:
    """
    Thread-safe LRU cache constrained by max_bytes (not max items).
    """
    def __init__(self, max_bytes: int = 512 * 1024 * 1024, weigher: Callable[[Any], int] = default_weigher):
        self.max_bytes = int(max_bytes)
        self.weigher = weigher
        self._od: "OrderedDict[Hashable, Tuple[Any, int]]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats = ByteCacheStats()

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            if key in self._od:
                self._od.move_to_end(key)
                self.stats.hits += 1
                return self._od[key][0]
            self.stats.misses += 1
            return None

    def put(self, key: Hashable, value: Any) -> None:
        w = int(self.weigher(value))

        with self._lock:
            if key in self._od:
                _, old_w = self._od[key]
                self.stats.bytes_in_use -= old_w
                self._od[key] = (value, w)
                self._od.move_to_end(key)
                self.stats.bytes_in_use += w
            else:
                self._od[key] = (value, w)
                self._od.move_to_end(key)
                self.stats.bytes_in_use += w

            # evict until within budget
            while self.stats.bytes_in_use > self.max_bytes and len(self._od) > 0:
                _, (_, ev_w) = self._od.popitem(last=False)
                self.stats.bytes_in_use -= ev_w
                self.stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._od.clear()
            self.stats = ByteCacheStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._od)
