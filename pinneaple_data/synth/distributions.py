from __future__ import annotations

from typing import List, Optional, Callable
import torch

from .base import SynthConfig, SynthOutput
from .pde import SimplePhysicalSample


class DistributionSynthGenerator:
    """
    Generate synthetic samples from distributions:
      - gaussian, uniform
      - mixture of gaussians (simple)
    Output as PhysicalSample-like objects: fields={"x":..., "y":...} etc.
    """
    def __init__(self, cfg: Optional[SynthConfig] = None):
        self.cfg = cfg or SynthConfig()

    def _rng(self):
        return torch.Generator(device="cpu").manual_seed(int(self.cfg.seed))

    def generate(
        self,
        *,
        kind: str = "gaussian",
        n_samples: int = 1024,
        dim: int = 2,
        mean: float = 0.0,
        std: float = 1.0,
        low: float = -1.0,
        high: float = 1.0,
        mixture_means: Optional[List[List[float]]] = None,
        mixture_stds: Optional[List[float]] = None,
        mixture_weights: Optional[List[float]] = None,
        y_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> SynthOutput:
        kind = kind.lower().strip()
        rng = self._rng()
        device = torch.device(self.cfg.device)
        dtype = getattr(torch, self.cfg.dtype)

        if kind == "gaussian":
            x = torch.randn((n_samples, dim), generator=rng, device=device, dtype=dtype) * float(std) + float(mean)
        elif kind == "uniform":
            x = (high - low) * torch.rand((n_samples, dim), generator=rng, device=device, dtype=dtype) + low
        elif kind == "mog":
            if not mixture_means or not mixture_stds:
                raise ValueError("mixture_means and mixture_stds required for mog")
            K = len(mixture_means)
            w = torch.tensor(mixture_weights or [1.0 / K] * K, device=device, dtype=dtype)
            w = w / w.sum()
            comp = torch.multinomial(w, num_samples=int(n_samples), replacement=True, generator=rng)
            means = torch.tensor(mixture_means, device=device, dtype=dtype)  # (K,dim)
            stds = torch.tensor(mixture_stds, device=device, dtype=dtype).view(K, 1)
            x = torch.randn((n_samples, dim), generator=rng, device=device, dtype=dtype) * stds[comp] + means[comp]
        else:
            raise ValueError("kind must be gaussian | uniform | mog")

        if y_fn is None:
            # default: nonlinear mapping to create supervised target
            y = (x ** 2).sum(dim=-1, keepdim=True)
        else:
            y = y_fn(x)

        sample = SimplePhysicalSample(
            fields={"x": x, "y": y},
            coords={},
            meta={"kind": kind, "dim": int(dim)},
        )
        return SynthOutput(samples=[sample], extras={"n_points": int(n_samples)})
