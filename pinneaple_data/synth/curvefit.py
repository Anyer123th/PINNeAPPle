from __future__ import annotations

from typing import Optional, Tuple
import torch

from .base import SynthConfig, SynthOutput
from .pde import SimplePhysicalSample


def _poly_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    # x: (N,1) -> (N, degree+1) [1, x, x^2, ...]
    feats = [torch.ones_like(x)]
    for d in range(1, degree + 1):
        feats.append(x ** d)
    return torch.cat(feats, dim=1)


class CurveFitSynthGenerator:
    """
    Learn a curve/trend from a dataset and synthesize missing values or densify sampling.

    MVP:
      - assumes you provide 1D input x and target y
      - fits polynomial regression (ridge)
      - generates new points and returns filled dataset
    """
    def __init__(self, cfg: Optional[SynthConfig] = None):
        self.cfg = cfg or SynthConfig()

    def generate(
        self,
        *,
        x: torch.Tensor,            # (N,1)
        y: torch.Tensor,            # (N,1)
        degree: int = 3,
        ridge: float = 1e-6,
        n_new: int = 1024,
        x_range: Optional[Tuple[float, float]] = None,
        noise_std: float = 0.0,
        mask_missing: Optional[torch.Tensor] = None,  # (N,1) bool; True means missing
    ) -> SynthOutput:
        device = torch.device(self.cfg.device)
        dtype = getattr(torch, self.cfg.dtype)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        if mask_missing is not None:
            m = (~mask_missing).view(-1)
            x_fit = x[m]
            y_fit = y[m]
        else:
            x_fit, y_fit = x, y

        Phi = _poly_features(x_fit, int(degree))  # (N,D)
        # ridge solve: (Phi^T Phi + λI) w = Phi^T y
        D = Phi.shape[1]
        I = torch.eye(D, device=device, dtype=dtype)
        w = torch.linalg.solve(Phi.t() @ Phi + float(ridge) * I, Phi.t() @ y_fit)

        # generate new x
        if x_range is None:
            lo = float(x.min().item())
            hi = float(x.max().item())
        else:
            lo, hi = float(x_range[0]), float(x_range[1])

        x_new = torch.linspace(lo, hi, int(n_new), device=device, dtype=dtype).view(-1, 1)
        Phi_new = _poly_features(x_new, int(degree))
        y_new = Phi_new @ w

        if noise_std and noise_std > 0:
            y_new = y_new + noise_std * torch.randn_like(y_new)

        sample = SimplePhysicalSample(
            fields={"x": x_new, "y": y_new, "w": w},
            coords={},
            meta={"degree": int(degree), "ridge": float(ridge), "source": "curvefit"},
        )
        return SynthOutput(samples=[sample], extras={"weights_shape": tuple(w.shape)})
