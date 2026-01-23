from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn as nn

from .base import AEBase


class Autoencoder2D(AEBase):
    """
    Conv2D autoencoder for images / 2D fields.

    Args:
      in_channels: channels in input (e.g. 1 for scalar field)
      latent_dim: latent vector dim
      img_size: (H,W) required to build final linear layers
      base_channels: width multiplier
    """
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        img_size: Tuple[int, int],
        base_channels: int = 32,
    ):
        super().__init__()
        H, W = img_size

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1), nn.GELU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            feat = self.enc(dummy)
            self._feat_shape = feat.shape[1:]  # (C,h,w)
            feat_dim = int(feat.numel())

        self.to_latent = nn.Linear(feat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, feat_dim)

        C, h, w = self._feat_shape
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(C, base_channels * 2, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(base_channels, in_channels, 4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        f = self.enc(x)
        f = f.view(x.shape[0], -1)
        return self.to_latent(f)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        f = self.from_latent(z)
        f = f.view(z.shape[0], *self._feat_shape)
        return self.dec(f)
