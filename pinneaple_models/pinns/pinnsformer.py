from __future__ import annotations

from typing import Dict, Optional, Callable, Any

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput


class PINNsFormer(PINNBase):
    """
    PINNsFormer (MVP):
      - sequence model for spatiotemporal tokens
      - x_seq: (B,T,in_dim)
      - outputs y_seq: (B,T,out_dim)

    You can use it for:
      - time tokens (t) + spatial embeddings
      - multi-sensor sequence
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        *,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> PINNOutput:
        h = self.in_proj(x_seq)                # (B,T,D)
        h = self.enc(h, mask=attn_mask)        # (B,T,D)
        y = self.out_proj(h)                   # (B,T,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if physics_fn is not None and physics_data is not None:
            pl = self.physics_loss(physics_fn=physics_fn, physics_data=physics_data)
            losses.update(pl)
            losses["total"] = losses["total"] + losses.get("physics", torch.tensor(0.0, device=y.device))
        return PINNOutput(y=y, losses=losses, extras={})
