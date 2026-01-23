from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .base import AEBase, AEOutput
from .dense_ae import _mlp


class VariationalAutoencoder(AEBase):
    """
    MLP VAE for vector inputs.

    Args:
      input_dim
      latent_dim
      hidden
      beta: weight on KL term (beta-VAE)
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden: List[int] = (512, 256),
        beta: float = 1.0,
        activation: str = "gelu",
    ):
        super().__init__()
        act = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(activation.lower(), nn.GELU())

        self.beta = float(beta)

        self.encoder = _mlp([input_dim, *list(hidden)], act, last_act=True)
        hdim = int(hidden[-1]) if hidden else input_dim

        self.mu = nn.Linear(hdim, latent_dim)
        self.logvar = nn.Linear(hdim, latent_dim)

        self.decoder = _mlp([latent_dim, *list(reversed(hidden)), input_dim], act, last_act=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # returns z sample (for base forward compatibility)
        x = x.view(x.shape[0], -1)
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self._reparam(mu, logvar)
        # stash for loss
        self._last_mu = mu
        self._last_logvar = logvar
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> AEOutput:
        z = self.encode(x)
        x_hat = self.decode(z)
        losses = self.loss_from_parts(x_hat=x_hat, z=z, x=x)
        extras = {"mu": self._last_mu, "logvar": self._last_logvar}
        return AEOutput(x_hat=x_hat, z=z, losses=losses, extras=extras)

    def _reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def loss_from_parts(self, *, x_hat: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.view(x.shape[0], -1)
        recon = torch.mean((x_hat - x) ** 2)

        mu = getattr(self, "_last_mu", None)
        logvar = getattr(self, "_last_logvar", None)
        if mu is None or logvar is None:
            kl = torch.tensor(0.0, device=x.device)
        else:
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

        total = recon + self.beta * kl
        return {"recon": recon, "kl": kl, "total": total}
