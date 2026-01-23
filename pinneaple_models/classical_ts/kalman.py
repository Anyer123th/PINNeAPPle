from __future__ import annotations
from typing import Dict, Optional, Any

import torch
import torch.nn as nn

from .base import ClassicalTSBase, ClassicalTSOutput


class KalmanFilter(ClassicalTSBase):
    """
    Linear Kalman Filter (batch, MVP):

      x_{t} = A x_{t-1} + B u_t + w,  w ~ N(0,Q)
      y_{t} = H x_{t}   + v,          v ~ N(0,R)

    Shapes:
      A: (n,n)
      H: (m,n)
      Q: (n,n)
      R: (m,m)
      B: (n,du) optional
    """
    def __init__(
        self,
        A: torch.Tensor,
        H: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        *,
        B: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("H", H)
        self.register_buffer("Q", Q)
        self.register_buffer("R", R)
        if B is not None:
            self.register_buffer("B", B)
        else:
            self.B = None

    def forward(
        self,
        y: torch.Tensor,                 # (B,T,m)
        *,
        u: Optional[torch.Tensor] = None, # (B,T,du)
        x0: Optional[torch.Tensor] = None, # (B,n)
        P0: Optional[torch.Tensor] = None, # (B,n,n) or (n,n)
        return_gain: bool = False,
    ) -> ClassicalTSOutput:
        Bsz, T, m = y.shape
        n = self.A.shape[0]
        device, dtype = y.device, y.dtype

        A = self.A.to(device=device, dtype=dtype)
        H = self.H.to(device=device, dtype=dtype)
        Q = self.Q.to(device=device, dtype=dtype)
        R = self.R.to(device=device, dtype=dtype)
        I = torch.eye(n, device=device, dtype=dtype).expand(Bsz, n, n)

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
            # predict
            if self.B is not None and u is not None:
                Bt = self.B.to(device=device, dtype=dtype)
                x = (A @ x.unsqueeze(-1)).squeeze(-1) + (Bt @ u[:, t, :].unsqueeze(-1)).squeeze(-1)
            else:
                x = (A @ x.unsqueeze(-1)).squeeze(-1)

            P = A @ P @ A.transpose(-1, -2) + Q

            # update
            yt = y[:, t, :]
            S = H @ P @ H.transpose(-1, -2) + R  # (B,m,m)
            K = P @ H.transpose(-1, -2) @ torch.linalg.inv(S)  # (B,n,m)

            innov = yt - (H @ x.unsqueeze(-1)).squeeze(-1)  # (B,m)
            x = x + (K @ innov.unsqueeze(-1)).squeeze(-1)
            P = (I - K @ H) @ P

            xs.append(x)
            Ps.append(P)
            if return_gain:
                Ks.append(K)

        x_filt = torch.stack(xs, dim=1)  # (B,T,n)
        extras: Dict[str, Any] = {"P": torch.stack(Ps, dim=1)}
        if return_gain:
            extras["K"] = torch.stack(Ks, dim=1)

        return ClassicalTSOutput(y=x_filt, losses={"total": torch.tensor(0.0, device=device)}, extras=extras)
