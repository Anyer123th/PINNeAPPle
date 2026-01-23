from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .base import AEBase
from .dense_ae import _mlp


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    # (n,d) (m,d) -> (n,m)
    x2 = (x**2).sum(dim=1, keepdim=True)
    y2 = (y**2).sum(dim=1, keepdim=True).t()
    dist2 = x2 - 2 * (x @ y.t()) + y2
    return torch.exp(-dist2 / (2 * sigma**2))


def _mmd_rbf(z: torch.Tensor, z_prior: torch.Tensor, sigma: float) -> torch.Tensor:
    kzz = _rbf_kernel(z, z, sigma)
    kpp = _rbf_kernel(z_prior, z_prior, sigma)
    kzp = _rbf_kernel(z, z_prior, sigma)
    return kzz.mean() + kpp.mean() - 2 * kzp.mean()


class KAEAutoencoder(AEBase):
    """
    Kernel Autoencoder (practical MVP):
      - Dense AE with an MMD penalty on latent space to match a simple prior.

    Args:
      input_dim, latent_dim, hidden
      mmd_weight: weight for MMD term
      mmd_sigma: RBF sigma
      prior: "normal" (N(0,1)) or "uniform" (-1,1)
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden: List[int] = (512, 256),
        activation: str = "gelu",
        mmd_weight: float = 1.0,
        mmd_sigma: float = 1.0,
        prior: str = "normal",
    ):
        super().__init__()
        act = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(activation.lower(), nn.GELU())
        self.mmd_weight = float(mmd_weight)
        self.mmd_sigma = float(mmd_sigma)
        self.prior = prior.lower().strip()

        enc_dims = [input_dim, *list(hidden), latent_dim]
        dec_dims = [latent_dim, *list(reversed(hidden)), input_dim]

        self.encoder = _mlp(enc_dims, act, last_act=False)
        self.decoder = _mlp(dec_dims, act, last_act=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def loss_from_parts(self, *, x_hat: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.view(x.shape[0], -1)
        recon = torch.mean((x_hat - x) ** 2)

        if self.prior == "uniform":
            z_prior = (2.0 * torch.rand_like(z) - 1.0)
        else:
            z_prior = torch.randn_like(z)

        mmd = _mmd_rbf(z, z_prior, sigma=self.mmd_sigma)
        total = recon + self.mmd_weight * mmd
        return {"recon": recon, "mmd": mmd, "total": total}
