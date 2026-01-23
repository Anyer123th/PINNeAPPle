from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ClassicalTSBase, ClassicalTSOutput


class VAR(ClassicalTSBase):
    """
    VAR(p) with ridge regression (closed form).

    x_t = c + sum_{k=1..p} A_k x_{t-k} + eps

    Fit uses batch-aggregated least squares over all sequences.
    """
    def __init__(self, dim: int, p: int = 1, l2: float = 1e-6, use_bias: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.p = int(p)
        self.l2 = float(l2)
        self.use_bias = bool(use_bias)

        feat_dim = self.dim * self.p + (1 if self.use_bias else 0)
        self.W = nn.Parameter(torch.zeros(feat_dim, self.dim), requires_grad=False)
        self._fitted = False

    def _make_design(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D) -> X: (B*(T-p), D*p (+1))
        B, T, D = x.shape
        rows = []
        for t in range(self.p, T):
            lagged = [x[:, t - k, :] for k in range(1, self.p + 1)]
            row = torch.cat(lagged, dim=-1)  # (B, D*p)
            if self.use_bias:
                row = torch.cat([row, torch.ones((B, 1), device=x.device, dtype=x.dtype)], dim=-1)
            rows.append(row)
        return torch.cat(rows, dim=0)

    def _targets(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,D) -> (B*(T-p), D)
        return torch.cat([x[:, t, :] for t in range(self.p, x.shape[1])], dim=0)

    @staticmethod
    def _ridge_solve(X: torch.Tensor, Y: torch.Tensor, l2: float) -> torch.Tensor:
        F = X.shape[1]
        I = torch.eye(F, device=X.device, dtype=X.dtype)
        return torch.linalg.solve(X.t() @ X + l2 * I, X.t() @ Y)

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "VAR":
        X = self._make_design(x)
        Y = self._targets(x)
        W = self._ridge_solve(X, Y, self.l2)
        self.W.copy_(W)
        self._fitted = True
        return self

    @torch.no_grad()
    def forecast(self, x_hist: torch.Tensor, steps: int) -> torch.Tensor:
        """
        x_hist: (B, T, D) with T >= p
        returns: (B, steps, D)
        """
        B, T, D = x_hist.shape
        if T < self.p:
            raise ValueError(f"Need at least p={self.p} history steps, got {T}.")
        buf = [x_hist[:, -k, :] for k in range(1, self.p + 1)]  # list of (B,D), newest first
        preds = []
        for _ in range(int(steps)):
            feat = torch.cat(buf, dim=-1)  # (B, D*p)
            if self.use_bias:
                feat = torch.cat([feat, torch.ones((B, 1), device=x_hist.device, dtype=x_hist.dtype)], dim=-1)
            y = feat @ self.W  # (B,D)
            preds.append(y)
            buf = [y] + buf[:-1]
        return torch.stack(preds, dim=1)

    def forward(self, x: torch.Tensor, *, y_true: Optional[torch.Tensor] = None, return_loss: bool = False) -> ClassicalTSOutput:
        # one-step ahead prediction over the provided sequence
        B, T, D = x.shape
        if T <= self.p:
            raise ValueError(f"T must be > p. Got T={T}, p={self.p}")

        X = self._make_design(x)  # (B*(T-p), F)
        Yhat = X @ self.W         # (B*(T-p), D)
        y_hat = Yhat.view(T - self.p, B, D).permute(1, 0, 2)  # (B, T-p, D)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = torch.mean((y_hat - y_true[:, self.p:, :]) ** 2)
            losses["total"] = losses["mse"]

        return ClassicalTSOutput(y=y_hat, losses=losses, extras={"fitted": self._fitted})
