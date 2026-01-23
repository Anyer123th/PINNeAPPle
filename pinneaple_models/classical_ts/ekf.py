from __future__ import annotations
from typing import Callable, Dict, Optional, Any

import torch

from .base import ClassicalTSBase, ClassicalTSOutput


class ExtendedKalmanFilter(ClassicalTSBase):
    """
    EKF (MVP):
      x_t = f(x_{t-1}, u_t) + w
      y_t = h(x_t) + v

    User provides:
      f: (x,u)->x
      h: x->y
      F_jac: Jacobian of f wrt x, shape (B,n,n)
      H_jac: Jacobian of h wrt x, shape (B,m,n)
    """
    def __init__(
        self,
        f: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        h: Callable[[torch.Tensor], torch.Tensor],
        Q: torch.Tensor,
        R: torch.Tensor,
        F_jac: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        H_jac: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.f = f
        self.h = h
        self.F_jac = F_jac
        self.H_jac = H_jac
        self.register_buffer("Q", Q)
        self.register_buffer("R", R)

    def forward(
        self,
        y: torch.Tensor,                  # (B,T,m)
        *,
        u: Optional[torch.Tensor] = None,  # (B,T,du)
        x0: Optional[torch.Tensor] = None, # (B,n)
        P0: Optional[torch.Tensor] = None, # (B,n,n) or (n,n)
        return_gain: bool = False,
    ) -> ClassicalTSOutput:
        Bsz, T, m = y.shape
        device, dtype = y.device, y.dtype

        Q = self.Q.to(device=device, dtype=dtype)
        R = self.R.to(device=device, dtype=dtype)

        if x0 is None:
            # infer n from Q
            n = Q.shape[0]
            x = torch.zeros((Bsz, n), device=device, dtype=dtype)
        else:
            x = x0.to(device=device, dtype=dtype)
            n = x.shape[1]

        if P0 is None:
            P = torch.eye(n, device=device, dtype=dtype).expand(Bsz, n, n).clone()
        else:
            P = P0.to(device=device, dtype=dtype)
            if P.ndim == 2:
                P = P.expand(Bsz, n, n).clone()

        I = torch.eye(n, device=device, dtype=dtype).expand(Bsz, n, n)

        xs, Ps, Ks = [], [], []
        for t in range(T):
            ut = None if u is None else u[:, t, :]

            # predict
            x = self.f(x, ut)
            Fm = self.F_jac(x, ut)  # (B,n,n)
            P = Fm @ P @ Fm.transpose(-1, -2) + Q

            # update
            yt = y[:, t, :]
            Hm = self.H_jac(x)      # (B,m,n)
            yhat = self.h(x)        # (B,m)

            S = Hm @ P @ Hm.transpose(-1, -2) + R
            K = P @ Hm.transpose(-1, -2) @ torch.linalg.inv(S)  # (B,n,m)

            innov = yt - yhat
            x = x + (K @ innov.unsqueeze(-1)).squeeze(-1)
            P = (I - K @ Hm) @ P

            xs.append(x)
            Ps.append(P)
            if return_gain:
                Ks.append(K)

        x_filt = torch.stack(xs, dim=1)
        extras: Dict[str, Any] = {"P": torch.stack(Ps, dim=1)}
        if return_gain:
            extras["K"] = torch.stack(Ks, dim=1)

        return ClassicalTSOutput(y=x_filt, losses={"total": torch.tensor(0.0, device=device)}, extras=extras)
