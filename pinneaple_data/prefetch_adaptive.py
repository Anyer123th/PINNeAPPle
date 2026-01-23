from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, List
import threading
import queue
import time

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .device import pin_sample, to_device_sample
from .zarr_cached_store_bytes import CachedUPDZarrStoreBytes, ZarrByteCacheConfig


@dataclass
class AdaptivePrefetchConfig:
    # Hard queue capacity (cannot change at runtime)
    queue_max: int = 64

    # Adaptive target fill bounds (producer tries to keep queue around target_fill)
    min_target_fill: int = 4
    max_target_fill: int = 32
    target_fill_init: int = 16

    # Control loop
    control_period_s: float = 0.5
    increase_step: int = 2
    decrease_step: int = 2

    # If consumer is slow (queue stays near full), decrease. If queue drains often, increase.
    high_watermark: float = 0.85   # qsize/queue_max
    low_watermark: float = 0.25

    # Caching + pin + device
    use_sample_cache: bool = True
    pin_memory: bool = True
    target_device: str = "cpu"     # "cpu" or "cuda"
    transfer_non_blocking: bool = True


class AdaptivePrefetchZarrUPDIterable(IterableDataset):
    """
    Adaptive prefetching:
      - queue has fixed max
      - controller adapts target_fill within [min,max] based on queue occupancy over time
    """
    def __init__(
        self,
        root: str,
        *,
        fields: Optional[Sequence[str]] = None,
        coords: Optional[Sequence[str]] = None,
        dtype: Optional[torch.dtype] = None,
        start: int = 0,
        end: Optional[int] = None,
        stride: int = 1,
        sample_ctor: Any = None,
        cache: Optional[ZarrByteCacheConfig] = None,
        cfg: Optional[AdaptivePrefetchConfig] = None,
    ):
        super().__init__()
        self.root = root
        self.fields = fields
        self.coords = coords
        self.dtype = dtype
        self.start = start
        self.end = end
        self.stride = stride
        self.sample_ctor = sample_ctor
        self.cache = cache or ZarrByteCacheConfig()
        self.cfg = cfg or AdaptivePrefetchConfig()

    def __iter__(self) -> Iterator[Any]:
        store = CachedUPDZarrStoreBytes(self.root, cache=self.cache, mode="r")
        n = store.count()
        end = n if self.end is None else min(self.end, n)

        wi = get_worker_info()
        if wi is None:
            indices = list(range(self.start, end, self.stride))
        else:
            worker_id = wi.id
            num_workers = wi.num_workers
            base = self.start + worker_id * self.stride
            step = num_workers * self.stride
            indices = list(range(base, end, step))

        q: "queue.Queue[Any]" = queue.Queue(maxsize=self.cfg.queue_max)
        stop = threading.Event()
        err_holder: List[BaseException] = []

        # Adaptive state (shared within worker process)
        target_fill = {"v": int(self.cfg.target_fill_init)}
        target_lock = threading.Lock()

        def clamp(x: int) -> int:
            return max(self.cfg.min_target_fill, min(self.cfg.max_target_fill, x))

        def controller():
            # Periodically adjust target_fill based on occupancy
            while not stop.is_set():
                time.sleep(self.cfg.control_period_s)
                occ = q.qsize() / float(self.cfg.queue_max)
                with target_lock:
                    cur = target_fill["v"]
                    if occ > self.cfg.high_watermark:
                        target_fill["v"] = clamp(cur - self.cfg.decrease_step)
                    elif occ < self.cfg.low_watermark:
                        target_fill["v"] = clamp(cur + self.cfg.increase_step)

        io_device = "cpu"

        def producer():
            try:
                for i in indices:
                    if stop.is_set():
                        break

                    # wait until queue is below current target_fill
                    while not stop.is_set():
                        with target_lock:
                            tf = target_fill["v"]
                        if q.qsize() < tf:
                            break
                        time.sleep(0.001)

                    s = store.read_sample(
                        i,
                        fields=self.fields,
                        coords=self.coords,
                        device=io_device,
                        dtype=self.dtype,
                        sample_ctor=self.sample_ctor,
                        use_sample_cache=self.cfg.use_sample_cache,
                    )
                    if self.cfg.pin_memory:
                        s = pin_sample(s)
                    q.put(s)

                q.put(None)
            except BaseException as e:
                err_holder.append(e)
                q.put(None)

        th_prod = threading.Thread(target=producer, daemon=True)
        th_ctrl = threading.Thread(target=controller, daemon=True)

        th_prod.start()
        th_ctrl.start()

        do_cuda = self.cfg.target_device.lower().startswith("cuda")

        while True:
            item = q.get()
            if item is None:
                break
            if err_holder:
                raise err_holder[0]
            if do_cuda:
                item = to_device_sample(item, device="cuda", dtype=self.dtype, non_blocking=self.cfg.transfer_non_blocking)
            yield item

        stop.set()
