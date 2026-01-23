from __future__ import annotations

from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class NeuralSDE(ContinuousModelBase):
    """
    Neural SDE (MVP, Euler–Maruyama):

      dY = f(t, y) dt + g(t, y) dW
    where g outputs diagonal diffusion by default.

    Inputs:
      y0: (B, state_dim)
      t:  (T,) increasing
    Output:
      y_path: (B, T, state_dim)

    Notes:
      - diffusion can be "diag" (default) or "full" (outputs a matrix)
      - stable training typically needs small dt; this is a clean scaffold.
    """
    def __init__(
        self,
        state_dim: int,
        *,
        hidden: int = 128,
        num_layers: int = 3,
        diffusion: Literal["diag", "full"] = "diag",
        activation: str = "tanh",
        min_log_sigma: float = -10.0,
        max_log_sigma: float = 2.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.diffusion = diffusion
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)

        act = (activation or "tanh").lower()
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(act, nn.Tanh)

        # drift f(t,y)
        f_layers = [nn.Linear(self.state_dim + 1, hidden), act_fn()]
        for _ in range(num_layers - 1):
            f_layers += [nn.Linear(hidden, hidden), act_fn()]
        f_layers += [nn.Linear(hidden, self.state_dim)]
        self.f = nn.Sequential(*f_layers)

        # diffusion g(t,y)
        if diffusion == "diag":
            out = self.state_dim
        else:
            out = self.state_dim * self.state_dim

        g_layers = [nn.Linear(self.state_dim + 1, hidden), act_fn()]
        for _ in range(num_layers - 1):
            g_layers += [nn.Linear(hidden, hidden), act_fn()]
        g_layers += [nn.Linear(hidden, out)]
        self.g = nn.Sequential(*g_layers)

    def _cat(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            tt = t.view(1, 1).expand(y.shape[0], 1)
        elif t.ndim == 1:
            tt = t[:, None]
        else:
            tt = t
        return torch.cat([tt.to(y.device, y.dtype), y], dim=-1)

    def drift(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.f(self._cat(t, y))

    def diffusion_term(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raw = self.g(self._cat(t, y))
        if self.diffusion == "diag":
            # treat raw as log_sigma, clamp for stability, then exp
            log_sigma = torch.clamp(raw, self.min_log_sigma, self.max_log_sigma)
            return torch.exp(log_sigma)  # (B,D)
        else:
            # full matrix (B, D, D)
            return raw.view(y.shape[0], self.state_dim, self.state_dim)

    def forward(
        self,
        y0: torch.Tensor,                 # (B,D)
        t: torch.Tensor,                  # (T,)
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,T,D)
        return_loss: bool = False,
    ) -> ContOutput:
        B, D = y0.shape
        if D != self.state_dim:
            raise ValueError(f"Expected y0 dim {self.state_dim}, got {D}")
        T = t.numel()

        ys = [y0]
        y = y0

        for i in range(T - 1):
            ti = t[i]
            dt = (t[i + 1] - t[i]).to(dtype=y.dtype, device=y.device)
            if dt.item() <= 0:
                raise ValueError("t must be strictly increasing for SDE integration.")

            f = self.drift(ti, y)  # (B,D)

            if self.diffusion == "diag":
                sigma = self.diffusion_term(ti, y)  # (B,D)
                dW = torch.randn_like(y) * torch.sqrt(dt)
                y = y + f * dt + sigma * dW
            else:
                G = self.diffusion_term(ti, y)  # (B,D,D)
                dW = torch.randn((B, D), device=y.device, dtype=y.dtype) * torch.sqrt(dt)
                noise = torch.einsum("bij,bj->bi", G, dW)
                y = y + f * dt + noise

            ys.append(y)

        y_path = torch.stack(ys, dim=1)  # (B,T,D)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_path.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y_path, y_true)
            losses["total"] = losses["mse"]

        return ContOutput(y=y_path, losses=losses, extras={"t": t})
