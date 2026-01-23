from __future__ import annotations
from typing import Callable, Dict, Optional, Any

import torch

from .base import ClassicalTSBase, ClassicalTSOutput


class UnscentedKalmanFilter(ClassicalTSBase):
    """
    UKF MVP (sigma-point filter), batch-friendly.

    User provides:
      f(x,u)->x
      h(x)->y

    Parameters:
      alpha, beta, kappa control sigma points.
    """
    def __init__(
        self,
        f: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        h: Callable[[torch.Tensor], torch.Tensor],
        Q: torch.Tensor,
        R: torch.Tensor,
        *,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        super().__init__()
        self.f = f
        self.h = h
        self.register_buffer("Q", Q)
        self.register_buffer("R", R)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

    def _sigma_points(self, x: torch.Tensor, P: torch.Tensor):
        # x: (B,n), P:(B,n,n)
        B, n = x.shape
        device, dtype = x.device, x.dtype

        lam = (self.alpha ** 2) * (n + self.kappa) - n
        c = n + lam

        # Cholesky
        S = torch.linalg.cholesky((c * P).to(dtype=dtype))

        pts = [x]
        for i in range(n):
            col = S[:, :, i]
            pts.append(x + col)
            pts.append(x - col)
        X = torch.stack(pts, dim=1)  # (B, 2n+1, n)

        Wm = torch.full((2 * n + 1,), 1.0 / (2.0 * c), device=device, dtype=dtype)
        Wc = Wm.clone()
        Wm[0] = lam / c
        Wc[0] = lam / c + (1.0 - self.alpha ** 2 + self.beta)

        return X, Wm, Wc

    def forward(
        self,
        y: torch.Tensor,                  # (B,T,m)
        *,
        u: Optional[torch.Tensor] = None,  # (B,T,du)
        x0: Optional[torch.Tensor] = None,
        P0: Optional[torch.Tensor] = None,
        return_gain: bool = False,
    ) -> ClassicalTSOutput:
        Bsz, T, m = y.shape
        device, dtype = y.device, y.dtype

        Q = self.Q.to(device=device, dtype=dtype)
        R = self.R.to(device=device, dtype=dtype)

        n = Q.shape[0]
        if x0 is None:
            x = torch.zeros((Bsz, n), device=device, dtype=dtype)
        else:
            x = x0.to(device=device, dtype=dtype)

        if P0 is None:
            P = torch.eye(n, device=device, dtype=dtype).expand(Bsz, n, n).clone()
        else:
            P = P0.to(device=device, dtype=dtype)
            if P.ndim == 2:
                P = P.expand(Bsz, n, n).clone()

        xs, Ps, Ks = [], [], []
        for t in range(T):
            ut = None if u is None else u[:, t, :]

            # sigma points
            X, Wm, Wc = self._sigma_points(x, P)  # (B,S,n)

            # predict through dynamics
            Xp = self.f(X.reshape(-1, n), None if ut is None else ut.repeat_interleave(X.shape[1], dim=0))
            Xp = Xp.view(Bsz, -1, n)

            x_pred = torch.sum(Xp * Wm[None, :, None], dim=1)  # (B,n)
            dX = Xp - x_pred[:, None, :]
            P_pred = torch.sum(Wc[None, :, None, None] * (dX[:, :, :, None] @ dX[:, :, None, :]), dim=1) + Q

            # predict observation
            Yp = self.h(Xp.reshape(-1, n)).view(Bsz, -1, m)
            y_pred = torch.sum(Yp * Wm[None, :, None], dim=1)  # (B,m)
            dY = Yp - y_pred[:, None, :]
            S = torch.sum(Wc[None, :, None, None] * (dY[:, :, :, None] @ dY[:, :, None, :]), dim=1) + R

            # cross-cov
            Pxy = torch.sum(Wc[None, :, None, None] * (dX[:, :, :, None] @ dY[:, :, None, :]), dim=1)  # (B,n,m)

            K = Pxy @ torch.linalg.inv(S)  # (B,n,m)
            innov = y[:, t, :] - y_pred
            x = x_pred + (K @ innov.unsqueeze(-1)).squeeze(-1)
            P = P_pred - K @ S @ K.transpose(-1, -2)

            xs.append(x)
            Ps.append(P)
            if return_gain:
                Ks.append(K)

        extras: Dict[str, Any] = {"P": torch.stack(Ps, dim=1)}
        if return_gain:
            extras["K"] = torch.stack(Ks, dim=1)

        return ClassicalTSOutput(y=torch.stack(xs, dim=1), losses={"total": torch.tensor(0.0, device=device)}, extras=extras)
