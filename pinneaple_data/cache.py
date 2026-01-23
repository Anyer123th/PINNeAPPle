from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Optional
from collections import OrderedDict
import threading


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class LRUCache:
    """
    Thread-safe LRU cache.

    Notes:
      - max_items is the primary control
      - you can add your own "weigher" if you want max_bytes later
    """
    def __init__(self, max_items: int = 256):
        self.max_items = int(max_items)
        self._od: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            if key in self._od:
                self._od.move_to_end(key)
                self.stats.hits += 1
                return self._od[key]
            self.stats.misses += 1
            return None

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if key in self._od:
                self._od[key] = value
                self._od.move_to_end(key)
                return
            self._od[key] = value
            self._od.move_to_end(key)

            while len(self._od) > self.max_items:
                self._od.popitem(last=False)
                self.stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._od.clear()
            self.stats = CacheStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._od)
