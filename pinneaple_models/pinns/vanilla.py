from __future__ import annotations

from typing import Dict, List, Optional, Callable, Any

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput


def _act(name: str) -> nn.Module:
    name = (name or "tanh").lower()
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}.get(name, nn.Tanh())


class VanillaPINN(PINNBase):
    """
    Standard fully-connected PINN for regression:
      y = f(x)

    Args:
      in_dim: number of input coordinates (e.g. t,x,y,z)
      out_dim: number of predicted fields (e.g. u,v,p)
      hidden: widths
      activation: tanh/relu/gelu/silu
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: List[int] = (128, 128, 128, 128),
        activation: str = "tanh",
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
        return PINNOutput(y=y, losses=losses, extras={})
