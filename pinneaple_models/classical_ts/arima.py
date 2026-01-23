from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ClassicalTSBase, ClassicalTSOutput


class ARIMA(ClassicalTSBase):
    """
    ARIMA(p,d,q) MVP without external deps:

    - Supports differencing order d
    - Supports AR(p) with ridge regression
    - q (MA) is NOT implemented in this MVP (kept for API compatibility)

    Use:
      fit(x) where x: (B,T,dim) or (B,T,1)
      forecast(x_hist, steps)
    """
    def __init__(self, dim: int = 1, p: int = 3, d: int = 0, q: int = 0, l2: float = 1e-6, use_bias: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.p = int(p)
        self.d = int(d)
        self.q = int(q)  # placeholder
        self.l2 = float(l2)
        self.use_bias = bool(use_bias)

        feat_dim = self.dim * self.p + (1 if self.use_bias else 0)
        self.W = nn.Parameter(torch.zeros(feat_dim, self.dim), requires_grad=False)
        self._fitted = False

    def _difference(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        y = x
        for _ in range(self.d):
            y = y[:, 1:, :] - y[:, :-1, :]
        return y

    def _undifference_forecast(self, x_hist: torch.Tensor, dx_fore: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct forecast in original space given differenced forecasts.
        For d>0, we cumulatively sum with last known values.
        """
        if self.d == 0:
            return dx_fore

        # base = last observed level(s)
        base = x_hist[:, -1, :]  # (B,D)
        y = []
        cur = base
        for t in range(dx_fore.shape[1]):
            cur = cur + dx_fore[:, t, :]
            y.append(cur)
        return torch.stack(y, dim=1)

    @staticmethod
    def _ridge_solve(X: torch.Tensor, Y: torch.Tensor, l2: float) -> torch.Tensor:
        F = X.shape[1]
        I = torch.eye(F, device=X.device, dtype=X.dtype)
        return torch.linalg.solve(X.t() @ X + l2 * I, X.t() @ Y)

    def _make_design(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        rows = []
        for t in range(self.p, T):
            lagged = [x[:, t - k, :] for k in range(1, self.p + 1)]
            row = torch.cat(lagged, dim=-1)
            if self.use_bias:
                row = torch.cat([row, torch.ones((B, 1), device=x.device, dtype=x.dtype)], dim=-1)
            rows.append(row)
        return torch.cat(rows, dim=0)

    def _targets(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, t, :] for t in range(self.p, x.shape[1])], dim=0)

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "ARIMA":
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x.shape[-1]}")
        xd = self._difference(x)
        if xd.shape[1] <= self.p:
            raise ValueError(f"Not enough timesteps after differencing: T'={xd.shape[1]} <= p={self.p}")

        X = self._make_design(xd)
        Y = self._targets(xd)
        self.W.copy_(self._ridge_solve(X, Y, self.l2))
        self._fitted = True
        return self

    @torch.no_grad()
    def forecast(self, x_hist: torch.Tensor, steps: int) -> torch.Tensor:
        if x_hist.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x_hist.shape[-1]}")
        xd = self._difference(x_hist)
        B, T, D = xd.shape
        if T < self.p:
            raise ValueError(f"Need at least p={self.p} points after differencing, got {T}.")

        buf = [xd[:, -k, :] for k in range(1, self.p + 1)]
        dx_preds = []
        for _ in range(int(steps)):
            feat = torch.cat(buf, dim=-1)
            if self.use_bias:
                feat = torch.cat([feat, torch.ones((B, 1), device=x_hist.device, dtype=x_hist.dtype)], dim=-1)
            dx = feat @ self.W
            dx_preds.append(dx)
            buf = [dx] + buf[:-1]
        dx_fore = torch.stack(dx_preds, dim=1)  # (B,steps,D)

        # undifference back to original
        return self._undifference_forecast(x_hist, dx_fore)

    def forward(self, x: torch.Tensor, *, y_true: Optional[torch.Tensor] = None, return_loss: bool = False) -> ClassicalTSOutput:
        # one-step-ahead predictions on provided x
        xd = self._difference(x)
        B, T, D = xd.shape
        X = self._make_design(xd)
        Yhat = X @ self.W
        yhat_d = Yhat.view(T - self.p, B, D).permute(1, 0, 2)  # differenced space

        # map to original for comparison (approx): undifference using rolling base
        # for d>0, we'll compare on differenced space by default to keep it correct.
        y_out = yhat_d

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            yd_true = self._difference(y_true)
            losses["mse"] = torch.mean((yhat_d - yd_true[:, self.p:, :]) ** 2)
            losses["total"] = losses["mse"]

        return ClassicalTSOutput(y=y_out, losses=losses, extras={"fitted": self._fitted, "space": "differenced"})
