from __future__ import annotations

from typing import Dict, Optional, Callable, Any

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput


class PIELM(PINNBase):
    """
    Physics-Informed Extreme Learning Machine (MVP).

    Structure:
      - Random (frozen) hidden layer: h = phi(Wx + b)
      - Trainable linear output: y = h @ Beta

    Supports:
      - standard forward
      - optional closed-form ridge fit via .fit_ridge(x, y)

    Notes:
      - Physics terms can be applied via physics_fn (autograd works through Beta and through x if needed).
      - W,b are frozen by default for ELM behavior.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        activation: str = "tanh",
        freeze_random: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)

        act = (activation or "tanh").lower()
        self.phi = {"tanh": torch.tanh, "relu": torch.relu, "gelu": torch.nn.functional.gelu, "silu": torch.nn.functional.silu}.get(act, torch.tanh)

        self.W = nn.Parameter(torch.randn(hidden_dim, in_dim) * (1.0 / (in_dim ** 0.5)))
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.Beta = nn.Parameter(torch.zeros(hidden_dim, out_dim))

        if freeze_random:
            self.W.requires_grad_(False)
            self.b.requires_grad_(False)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,in_dim) -> h: (N,hidden_dim)
        h = x @ self.W.t() + self.b[None, :]
        return self.phi(h)

    @torch.no_grad()
    def fit_ridge(self, x: torch.Tensor, y: torch.Tensor, l2: float = 1e-6) -> None:
        """
        Closed-form ridge regression:
          Beta = (H^T H + l2 I)^-1 H^T Y
        """
        H = self.hidden(x)  # (N,H)
        HT = H.t()
        A = HT @ H
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        Beta = torch.linalg.solve(A + l2 * I, HT @ y)
        self.Beta.copy_(Beta)

    def forward(
        self,
        x: torch.Tensor,
        *,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
    ) -> PINNOutput:
        h = self.hidden(x)
        y = h @ self.Beta
        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if physics_fn is not None and physics_data is not None:
            pl = self.physics_loss(physics_fn=physics_fn, physics_data=physics_data)
            losses.update(pl)
            losses["total"] = losses["total"] + losses.get("physics", torch.tensor(0.0, device=y.device))
        return PINNOutput(y=y, losses=losses, extras={"hidden": h})
