from __future__ import annotations

from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class NeuralCDE(ContinuousModelBase):
    """
    Neural CDE (MVP, no torchcde):

    We treat x(t) as a control path and evolve hidden state h(t) as:
        dh = f(h,t) dt + G(h,t) dX
    with piecewise-linear interpolation of X, discretized per interval:
        h_{i+1} = h_i + f(h_i,t_i) * dt + (G(h_i,t_i) @ dX_i)

    Inputs:
      x: (B, T, input_dim) control observations
      t: (T,) strictly increasing
    Output:
      y_hat: (B, T, out_dim) (same timestamps)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        f_hidden: int = 128,
        g_hidden: int = 128,
        num_layers: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)

        act = (activation or "tanh").lower()
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(act, nn.Tanh)

        # initial hidden from first observation
        self.h0_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            act_fn(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # drift f: (h, t) -> (hidden_dim)
        f_layers = [nn.Linear(self.hidden_dim + 1, f_hidden), act_fn()]
        for _ in range(num_layers - 1):
            f_layers += [nn.Linear(f_hidden, f_hidden), act_fn()]
        f_layers += [nn.Linear(f_hidden, self.hidden_dim)]
        self.f = nn.Sequential(*f_layers)

        # control effect G: (h, t) -> (hidden_dim, input_dim)
        g_layers = [nn.Linear(self.hidden_dim + 1, g_hidden), act_fn()]
        for _ in range(num_layers - 1):
            g_layers += [nn.Linear(g_hidden, g_hidden), act_fn()]
        g_layers += [nn.Linear(g_hidden, self.hidden_dim * self.input_dim)]
        self.G = nn.Sequential(*g_layers)

        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def _ft(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # h: (B,H), t: scalar or (B,1)
        if t.ndim == 0:
            tt = t.view(1, 1).expand(h.shape[0], 1)
        elif t.ndim == 1:
            tt = t[:, None]
        else:
            tt = t
        inp = torch.cat([h, tt.to(h.device, h.dtype)], dim=-1)
        return self.f(inp)

    def _Gt(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            tt = t.view(1, 1).expand(h.shape[0], 1)
        elif t.ndim == 1:
            tt = t[:, None]
        else:
            tt = t
        inp = torch.cat([h, tt.to(h.device, h.dtype)], dim=-1)
        G = self.G(inp)  # (B, H*D)
        return G.view(h.shape[0], self.hidden_dim, self.input_dim)  # (B,H,D)

    def forward(
        self,
        x: torch.Tensor,            # (B,T,input_dim)
        t: torch.Tensor,            # (T,)
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,T,out_dim)
        return_loss: bool = False,
    ) -> ContOutput:
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected x.shape[-1]=={self.input_dim}, got {D}")
        if t.numel() != T:
            raise ValueError(f"Expected t length {T}, got {t.numel()}")

        h = self.h0_net(x[:, 0, :])  # (B,H)
        ys = [self.readout(h)]

        for i in range(T - 1):
            dt = (t[i + 1] - t[i]).to(dtype=h.dtype, device=h.device)
            dX = (x[:, i + 1, :] - x[:, i, :])  # (B,D)

            fval = self._ft(h, t[i])
            Gval = self._Gt(h, t[i])            # (B,H,D)
            control = torch.einsum("bhd,bd->bh", Gval, dX)

            h = h + fval * dt + control
            ys.append(self.readout(h))

        y_hat = torch.stack(ys, dim=1)  # (B,T,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y_hat, y_true)
            losses["total"] = losses["mse"]

        return ContOutput(y=y_hat, losses=losses, extras={"h_last": h})
