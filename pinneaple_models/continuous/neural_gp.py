from __future__ import annotations

from typing import Dict, Optional, Literal, Tuple

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class _RandomFourierFeatures(nn.Module):
    """
    RBF kernel approximation with random Fourier features:
      k(x,x') ~ phi(x)^T phi(x')

    phi(x) = sqrt(2/m) * cos(Wx + b), W ~ N(0, 1/l^2), b ~ U(0,2pi)
    """
    def __init__(self, in_dim: int, num_features: int = 512, lengthscale: float = 1.0, freeze: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.m = int(num_features)
        self.lengthscale = float(lengthscale)

        W = torch.randn(self.m, self.in_dim) * (1.0 / max(self.lengthscale, 1e-12))
        b = torch.rand(self.m) * (2.0 * torch.pi)

        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)

        if freeze:
            self.W.requires_grad_(False)
            self.b.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) -> (..., m)
        proj = x @ self.W.t() + self.b
        return (2.0 / self.m) ** 0.5 * torch.cos(proj)


class NeuralGaussianProcess(ContinuousModelBase):
    """
    Neural Gaussian Process (MVP):

    - Feature extractor (MLP) -> embedding
    - Random Fourier Features (RBF approx) on embedding
    - Linear Gaussian head for mean + (optional) homoscedastic noise

    This is not a full GP posterior (no matrix inversion per batch),
    but gives GP-like inductive bias + uncertainty proxy.

    Output:
      mu: (B,T,out_dim) or (B,N,out_dim)
      extras["logvar"]: (same shape) using learned noise + feature magnitude proxy
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        embed_dim: int = 64,
        mlp_hidden: int = 128,
        mlp_layers: int = 2,
        rff_features: int = 512,
        rff_lengthscale: float = 1.0,
        freeze_rff: bool = True,
        noise_mode: Literal["learned", "fixed"] = "learned",
        fixed_noise: float = 1e-3,
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.noise_mode = noise_mode
        self.fixed_noise = float(fixed_noise)

        # MLP feature extractor
        layers = [nn.Linear(self.in_dim, mlp_hidden), nn.GELU()]
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.GELU()]
        layers += [nn.Linear(mlp_hidden, embed_dim)]
        self.phi_nn = nn.Sequential(*layers)

        self.rff = _RandomFourierFeatures(embed_dim, num_features=rff_features, lengthscale=rff_lengthscale, freeze=freeze_rff)
        self.head = nn.Linear(rff_features, out_dim, bias=True)

        if noise_mode == "learned":
            self.log_noise = nn.Parameter(torch.tensor(-6.0))  # exp(-6) ~ 0.0025
        else:
            self.register_buffer("log_noise_buf", torch.tensor(torch.log(torch.tensor(self.fixed_noise))))

    def _flatten(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        return x2, shape[:-1]

    def forward(
        self,
        x: torch.Tensor,  # (..., in_dim)
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        use_nll: bool = True,
    ) -> ContOutput:
        x2, prefix = self._flatten(x)

        emb = self.phi_nn(x2)
        feat = self.rff(emb)
        mu2 = self.head(feat)  # (-1,out_dim)

        # very simple predictive variance proxy
        if self.noise_mode == "learned":
            base_logvar = self.log_noise
        else:
            base_logvar = self.log_noise_buf

        # add a tiny dependence on feature magnitude to avoid constant var everywhere
        mag = torch.mean(feat.pow(2), dim=-1, keepdim=True).clamp_min(1e-12)
        logvar2 = base_logvar + torch.log(mag)

        mu = mu2.reshape(*prefix, self.out_dim)
        logvar = logvar2.reshape(*prefix, 1).expand(*prefix, self.out_dim)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            if use_nll:
                losses["nll"] = self.gaussian_nll(mu, logvar, y_true)
                losses["total"] = losses["nll"]
            else:
                losses["mse"] = self.mse(mu, y_true)
                losses["total"] = losses["mse"]

        return ContOutput(y=mu, losses=losses, extras={"logvar": logvar})
