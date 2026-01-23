from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput
from .neural_ode import NeuralODE


class LatentODE(ContinuousModelBase):
    """
    Latent ODE MVP:
      - Encoder maps observed sequence -> latent z0 (mu, logvar)
      - Decoder integrates z(t) with NeuralODE then maps to x(t)
    """
    def __init__(self, obs_dim: int, latent_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)

        self.encoder_rnn = nn.GRU(obs_dim, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        self.ode = NeuralODE(state_dim=latent_dim, hidden=hidden, num_layers=2, method="rk4")
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, obs_dim))

    def _reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(
        self,
        x: torch.Tensor,  # (B,T,obs_dim)
        t: torch.Tensor,  # (T,)
        *,
        y_true: Optional[torch.Tensor] = None,
        beta_kl: float = 1.0,
        return_loss: bool = False,
    ) -> ContOutput:
        B, T, D = x.shape
        _, h = self.encoder_rnn(x)
        h = h[-1]

        mu = self.mu(h)
        logvar = self.logvar(h)
        z0 = self._reparam(mu, logvar)

        z_path = self.ode(z0, t).y  # (B,T,latent_dim)
        y_hat = self.decoder(z_path)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            rec = self.mse(y_hat, y_true)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            losses["rec"] = rec
            losses["kl"] = kl
            losses["total"] = rec + float(beta_kl) * kl

        return ContOutput(y=y_hat, losses=losses, extras={"mu": mu, "logvar": logvar})
