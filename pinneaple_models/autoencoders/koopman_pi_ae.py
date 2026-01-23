from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .base import AEBase, AEOutput
from .dense_ae import _mlp


class PhysicsInformedKoopmanAutoencoder(AEBase):
    """
    Koopman AE (MVP):
      - Dense AE
      - Trainable linear Koopman operator K in latent space
      - Optional dynamic consistency loss if x_next is provided

    forward supports:
      - forward(x) standard
      - forward(x, x_next=...) to add koopman loss

    Args:
      input_dim, latent_dim, hidden
      koopman_weight: weight of ||z_next - K z||^2
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden: List[int] = (512, 256),
        activation: str = "gelu",
        koopman_weight: float = 1.0,
    ):
        super().__init__()
        act = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(activation.lower(), nn.GELU())
        self.koopman_weight = float(koopman_weight)

        self.encoder = _mlp([input_dim, *list(hidden), latent_dim], act, last_act=False)
        self.decoder = _mlp([latent_dim, *list(reversed(hidden)), input_dim], act, last_act=False)

        # Koopman operator (latent_dim x latent_dim)
        self.K = nn.Parameter(torch.eye(latent_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, *, x_next: Optional[torch.Tensor] = None) -> AEOutput:
        z = self.encode(x)
        x_hat = self.decode(z)

        losses = self.loss_from_parts(x_hat=x_hat, z=z, x=x)

        extras: Dict[str, torch.Tensor] = {"K": self.K}

        if x_next is not None:
            z_next = self.encode(x_next)
            z_pred = z @ self.K.t()
            koop = torch.mean((z_next - z_pred) ** 2)
            losses["koopman"] = koop
            losses["total"] = losses["total"] + self.koopman_weight * koop
            extras["z_next"] = z_next
            extras["z_pred"] = z_pred

        return AEOutput(x_hat=x_hat, z=z, losses=losses, extras=extras)
