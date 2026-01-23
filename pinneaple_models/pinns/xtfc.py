from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any, Literal, Tuple

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput


@dataclass
class XTFCConfig:
    in_dim: int
    out_dim: int
    rf_dim: int = 2048
    activation: str = "tanh"
    freeze_random: bool = True

    # random features
    rf_kind: Literal["rff", "random_linear"] = "rff"   # rff = random Fourier features
    rff_sigma: float = 1.0                            # lengthscale (larger => smoother)
    use_bias: bool = True
    init_scale: float = 1.0                           # scales W initialization

    # stabilization / training
    head_bias: bool = True
    clamp_B: Optional[Tuple[float, float]] = None     # e.g. (0.0, 1.0) if you want
    eps_B: float = 0.0                                # add epsilon to avoid zero gradient zones

    # optional regularization hooks (returned in losses dict)
    l2_head: float = 0.0
    l2_W: float = 0.0

    # output transform
    broadcast_B: bool = True                          # if B returns (N,1) broadcast to (N,out_dim)


def _get_phi(name: str):
    name = (name or "tanh").lower()
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "gelu":
        return torch.nn.functional.gelu
    if name == "silu":
        return torch.nn.functional.silu
    return torch.tanh


class XTFC(PINNBase):
    """
    XTFC (Extreme Theory of Functional Connections) — improved MVP.

    Core construction:
      y(x) = g(x) + B(x) * N(x)

    Where:
      - g(x) satisfies constraints exactly (IC/BC)
      - B(x) vanishes on constraint boundaries (or enforces hard structure)
      - N(x) is a flexible approximator (extreme = random features + linear head)

    Improvements vs the prior scaffold:
      ✅ Config dataclass for reproducibility
      ✅ Random Fourier Features option (rf_kind="rff") — usually better behaved than pure random linear
      ✅ Optional regularization terms (l2_head, l2_W) returned in losses
      ✅ Robust shape handling for g(x) / B(x) + broadcasting
      ✅ Optional clamping / epsilon for B to avoid dead gradients and numerical issues
      ✅ Deterministic re-seeding via `reset_random_features(seed=...)`
      ✅ Optional `fit_ridge(...)` for fast closed-form head training (typical “extreme” workflow)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rf_dim: int = 2048,
        activation: str = "tanh",
        freeze_random: bool = True,
        g_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        B_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        *,
        config: Optional[XTFCConfig] = None,
    ):
        super().__init__()

        self.cfg = config or XTFCConfig(
            in_dim=in_dim,
            out_dim=out_dim,
            rf_dim=rf_dim,
            activation=activation,
            freeze_random=freeze_random,
        )

        self.in_dim = int(self.cfg.in_dim)
        self.out_dim = int(self.cfg.out_dim)
        self.rf_dim = int(self.cfg.rf_dim)

        self.phi = _get_phi(self.cfg.activation)

        # Random features parameters
        # For RFF:
        #   phi(x) = [cos(Wx + b), sin(Wx + b)] -> dimension 2*rf_dim
        # We'll store W of shape (rf_dim, in_dim) and b of shape (rf_dim)
        self.W = nn.Parameter(torch.empty(self.rf_dim, self.in_dim))
        self.b = nn.Parameter(torch.empty(self.rf_dim)) if self.cfg.use_bias else None

        self._init_random_features()

        # Linear head
        head_in = self._rf_out_dim()
        self.head = nn.Linear(head_in, self.out_dim, bias=bool(self.cfg.head_bias))

        if self.cfg.freeze_random:
            self.W.requires_grad_(False)
            if self.b is not None:
                self.b.requires_grad_(False)

        # user constraint constructors
        self.g_fn = g_fn
        self.B_fn = B_fn

    # -------------------------
    # Random features
    # -------------------------

    def _rf_out_dim(self) -> int:
        return 2 * self.rf_dim if self.cfg.rf_kind == "rff" else self.rf_dim

    def _init_random_features(self):
        # W init
        scale = float(self.cfg.init_scale)
        # For RFF: W ~ N(0, 1/sigma^2)
        if self.cfg.rf_kind == "rff":
            sigma = float(self.cfg.rff_sigma)
            std = (1.0 / max(sigma, 1e-12))
            nn.init.normal_(self.W, mean=0.0, std=std * scale)
            if self.b is not None:
                # b uniform in [0, 2π]
                self.b.data.uniform_(0.0, 2.0 * torch.pi)
        else:
            # random_linear: W ~ N(0, 1/sqrt(in_dim))
            std = (1.0 / max(self.in_dim, 1) ** 0.5)
            nn.init.normal_(self.W, mean=0.0, std=std * scale)
            if self.b is not None:
                nn.init.zeros_(self.b)

    @torch.no_grad()
    def reset_random_features(self, seed: Optional[int] = None) -> None:
        """
        Re-sample random features (useful for ensembles / retries).
        """
        if seed is not None:
            g = torch.Generator(device=self.W.device)
            g.manual_seed(int(seed))
            # manual init with generator
            scale = float(self.cfg.init_scale)
            if self.cfg.rf_kind == "rff":
                sigma = float(self.cfg.rff_sigma)
                std = (1.0 / max(sigma, 1e-12))
                self.W.copy_(torch.randn(self.rf_dim, self.in_dim, generator=g, device=self.W.device) * (std * scale))
                if self.b is not None:
                    self.b.copy_(torch.rand(self.rf_dim, generator=g, device=self.W.device) * (2.0 * torch.pi))
            else:
                std = (1.0 / max(self.in_dim, 1) ** 0.5)
                self.W.copy_(torch.randn(self.rf_dim, self.in_dim, generator=g, device=self.W.device) * (std * scale))
                if self.b is not None:
                    self.b.zero_()
        else:
            self._init_random_features()

    def _rf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Random features:
          - rff: concat[cos(Wx+b), sin(Wx+b)]
          - random_linear: phi(Wx+b) with chosen activation
        """
        proj = x @ self.W.t()
        if self.b is not None:
            proj = proj + self.b[None, :]

        if self.cfg.rf_kind == "rff":
            return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        else:
            return self.phi(proj)

    # -------------------------
    # Optional extreme solver
    # -------------------------

    @torch.no_grad()
    def fit_ridge(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        *,
        l2: float = 1e-6,
    ) -> None:
        """
        Closed-form ridge fit for the head only (extreme learning style).

        We fit in the *unconstrained* space:
          y_target should be the desired residual term for y = g + B*N
        Typical workflow:
          1) compute g(x), B(x)
          2) set y_target = (y_true - g) / B   (careful at boundary where B=0)
          3) fit head using features rf(x)

        NOTE: This does not update W/b (random features).
        """
        H = self._rf(x)  # (N, F)
        Y = y_target     # (N, out_dim)

        HT = H.t()
        A = HT @ H
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        Beta = torch.linalg.solve(A + l2 * I, HT @ Y)  # (F, out_dim)

        # assign to head weight (out_dim, F) in Linear
        self.head.weight.copy_(Beta.t())
        if self.head.bias is not None:
            self.head.bias.zero_()

    # -------------------------
    # Constraint pieces
    # -------------------------

    def _compute_g(self, x: torch.Tensor) -> torch.Tensor:
        if self.g_fn is None:
            return torch.zeros((x.shape[0], self.out_dim), device=x.device, dtype=x.dtype)
        g = self.g_fn(x)
        if g.ndim == 1:
            g = g[:, None]
        if g.shape[-1] == 1 and self.out_dim > 1:
            g = g.expand(-1, self.out_dim)
        if g.shape[-1] != self.out_dim:
            raise ValueError(f"g_fn(x) must output shape (N,{self.out_dim}) (or broadcastable), got {tuple(g.shape)}")
        return g

    def _compute_B(self, x: torch.Tensor) -> torch.Tensor:
        if self.B_fn is None:
            B = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        else:
            B = self.B_fn(x)
            if B.ndim == 1:
                B = B[:, None]

        # optional eps/clamp
        if self.cfg.eps_B != 0.0:
            B = B + float(self.cfg.eps_B)

        if self.cfg.clamp_B is not None:
            lo, hi = self.cfg.clamp_B
            B = torch.clamp(B, float(lo), float(hi))

        # broadcasting behavior
        if self.cfg.broadcast_B and B.shape[-1] == 1 and self.out_dim > 1:
            B = B.expand(-1, self.out_dim)

        if B.shape[-1] not in (1, self.out_dim):
            raise ValueError(f"B_fn(x) must output shape (N,1) or (N,{self.out_dim}), got {tuple(B.shape)}")

        return B

    # -------------------------
    # Forward
    # -------------------------

    def forward(
        self,
        x: torch.Tensor,
        *,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
        return_parts: bool = True,
    ) -> PINNOutput:
        # Random features -> head
        rf = self._rf(x)
        n = self.head(rf)  # N(x)

        g = self._compute_g(x)
        B = self._compute_B(x)

        # y = g + B * N
        y = g + B * n

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}

        # Regularization hooks (optional)
        reg = torch.tensor(0.0, device=y.device)
        if self.cfg.l2_head > 0.0:
            reg = reg + float(self.cfg.l2_head) * (self.head.weight.pow(2).mean())
        if self.cfg.l2_W > 0.0:
            reg = reg + float(self.cfg.l2_W) * (self.W.pow(2).mean())
        if reg.item() != 0.0:
            losses["reg"] = reg
            losses["total"] = losses["total"] + reg

        # Physics hook
        if physics_fn is not None and physics_data is not None:
            pl = self.physics_loss(physics_fn=physics_fn, physics_data=physics_data)
            losses.update(pl)
            losses["total"] = losses["total"] + losses.get("physics", torch.tensor(0.0, device=y.device))

        extras = {}
        if return_parts:
            extras = {"g": g, "B": B, "N": n, "rf": rf}

        return PINNOutput(y=y, losses=losses, extras=extras)
