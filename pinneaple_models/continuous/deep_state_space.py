from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class DeepStateSpaceModel(ContinuousModelBase):
    """
    Deep State Space Model (DSSM) MVP:

    Latent dynamics:
      z_t = RNN(z_{t-1}, x_t)
    Emission:
      y_t ~ N(mu(z_t), diag(exp(logvar(z_t))))

    This is a strong baseline for probabilistic sequences without external libs.

    Inputs:
      x: (B,T,input_dim)  (can be exogenous inputs; if none, pass zeros)
    Output:
      y_hat (mu): (B,T,out_dim), with extras["logvar"]
    """
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        latent_dim: int = 128,
        num_layers: int = 1,
        cell: str = "gru",   # "gru" or "lstm"
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.latent_dim = int(latent_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        cell = (cell or "gru").lower().strip()
        if cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")
        self.cell = cell

        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=int(num_layers),
            batch_first=True,
        )

        self.mu = nn.Linear(latent_dim, out_dim)
        self.logvar = nn.Linear(latent_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,  # (B,T,input_dim)
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,T,out_dim)
        return_loss: bool = False,
        use_nll: bool = True,
    ) -> ContOutput:
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected x dim {self.input_dim}, got {D}")

        h, _ = self.rnn(x)  # (B,T,latent_dim)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), self.min_logvar, self.max_logvar)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            if use_nll:
                losses["nll"] = self.gaussian_nll(mu, logvar, y_true)
                losses["total"] = losses["nll"]
            else:
                losses["mse"] = self.mse(mu, y_true)
                losses["total"] = losses["mse"]

        return ContOutput(y=mu, losses=losses, extras={"logvar": logvar, "latent": h})
