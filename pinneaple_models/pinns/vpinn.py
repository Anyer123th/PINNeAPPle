from __future__ import annotations

from typing import Dict, Optional, Callable, Any

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput
from .vanilla import _act


class VPINN(PINNBase):
    """
    Variational PINN (MVP scaffold).

    Key idea:
      - Instead of enforcing strong-form residual at points,
        you enforce weak form:
          ∫_Ω R(u) * v_i dΩ = 0  for a set of test functions v_i

    MVP design:
      - Model is a standard MLP.
      - Provide a `weak_fn(model, quad)` callable that returns a dict of weak losses.

    Inputs:
      x: (N,in_dim) collocation points (quadrature)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden=(128, 128, 128),
        activation: str = "tanh",
    ):
        super().__init__()
        act = _act(activation)
        dims = [in_dim, *list(hidden), out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        *,
        weak_fn: Optional[Callable[..., Dict[str, torch.Tensor]]] = None,
        weak_data: Optional[Dict[str, Any]] = None,
    ) -> PINNOutput:
        y = self.net(x)
        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}

        if weak_fn is not None and weak_data is not None:
            weak_losses = weak_fn(self, weak_data)
            if not isinstance(weak_losses, dict):
                raise TypeError("weak_fn must return a dict[str, torch.Tensor]")
            losses.update({f"weak/{k}": v for k, v in weak_losses.items()})
            total = sum(v for v in weak_losses.values())
            losses["weak"] = total
            losses["total"] = losses["total"] + total

        return PINNOutput(y=y, losses=losses, extras={})
