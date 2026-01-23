from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import torch


@dataclass
class ChunkSpec:
    """
    Chunking strategy for large tensors.

    mode:
      - "none"
      - "time": assumes tensor shape (T, ...)
      - "spatial": assumes tensor shape (..., N) or (T, N)
    """
    mode: str = "none"
    chunk_size: int = 1024
    dim: int = 0  # default time dim

def iter_chunks(x: torch.Tensor, spec: ChunkSpec) -> Iterator[torch.Tensor]:
    if spec.mode == "none":
        yield x
        return
    d = spec.dim
    n = x.shape[d]
    cs = int(spec.chunk_size)
    for i in range(0, n, cs):
        sl = [slice(None)] * x.ndim
        sl[d] = slice(i, min(i + cs, n))
        yield x[tuple(sl)]
