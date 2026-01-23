from __future__ import annotations

from typing import Dict, List, Optional, Callable, Any

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput
from .vanilla import _act


class InversePINN(PINNBase):
    """
    PINN with trainable inverse parameters.

    Example:
      inverse_params=["nu", "rho"]
      initial_guesses={"nu":1e-3, "rho":1.0}

    The inverse params are exposed as `self.inverse_params`.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: List[int] = (128, 128, 128, 128),
        activation: str = "tanh",
        inverse_params: Optional[List[str]] = None,
        initial_guesses: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        act = _act(activation)

        dims = [in_dim, *list(hidden), out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.net = nn.Sequential(*layers)

        self.inverse_params = nn.ParameterDict()
        if inverse_params:
            initial_guesses = initial_guesses or {}
            for name in inverse_params:
                v0 = float(initial_guesses.get(name, 0.1))
                self.inverse_params[name] = nn.Parameter(torch.tensor(v0, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        *,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
    ) -> PINNOutput:
        y = self.net(x)
        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if physics_fn is not None and physics_data is not None:
            pl = self.physics_loss(physics_fn=physics_fn, physics_data=physics_data)
            losses.update(pl)
            losses["total"] = losses["total"] + losses.get("physics", torch.tensor(0.0, device=y.device))
        extras = {"inverse_params": {k: v for k, v in self.inverse_params.items()}}
        return PINNOutput(y=y, losses=losses, extras=extras)
