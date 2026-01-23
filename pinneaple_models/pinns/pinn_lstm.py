from __future__ import annotations

from typing import Dict, Optional, Callable, Any

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput


class PINNLSTM(PINNBase):
    """
    PINN-LSTM (MVP):
      - Input is a sequence of coords/features: x_seq (B,T,in_dim)
      - LSTM produces hidden states; head predicts y at each time step: (B,T,out_dim)

    Useful for time-series PDE/ODE settings where temporal memory helps.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        *,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
    ) -> PINNOutput:
        h, _ = self.lstm(x_seq)          # (B,T,H)
        y = self.head(h)                 # (B,T,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if physics_fn is not None and physics_data is not None:
            pl = self.physics_loss(physics_fn=physics_fn, physics_data=physics_data)
            losses.update(pl)
            losses["total"] = losses["total"] + losses.get("physics", torch.tensor(0.0, device=y.device))
        return PINNOutput(y=y, losses=losses, extras={})
