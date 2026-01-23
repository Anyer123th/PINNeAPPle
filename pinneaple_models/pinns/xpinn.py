from __future__ import annotations

from typing import Dict, List, Optional, Callable, Any, Tuple

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput
from .vanilla import VanillaPINN


class XPINN(PINNBase):
    """
    XPINN (MVP):
      - Multiple sub-PINNs, each responsible for a subdomain.
      - Interface loss enforces continuity between neighboring subdomains.

    Usage pattern (MVP):
      forward expects:
        x_list: list of tensors, one per subdomain: [x0, x1, ...]
      and optionally:
        interface_pairs: list of tuples (i, j, x_iface_i, x_iface_j)
          where x_iface_* are matching interface point coords.

    Physics:
      physics_fn can be applied per-subdomain by providing physics_data_list.
    """
    def __init__(
        self,
        n_subdomains: int,
        in_dim: int,
        out_dim: int,
        hidden=(128, 128, 128, 128),
        activation: str = "tanh",
        interface_weight: float = 1.0,
    ):
        super().__init__()
        self.interface_weight = float(interface_weight)
        self.subnets = nn.ModuleList([
            VanillaPINN(in_dim=in_dim, out_dim=out_dim, hidden=list(hidden), activation=activation)
            for _ in range(int(n_subdomains))
        ])

    def forward(
        self,
        x_list: List[torch.Tensor],
        *,
        interface_pairs: Optional[List[Tuple[int, int, torch.Tensor, torch.Tensor]]] = None,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data_list: Optional[List[Dict[str, Any]]] = None,
    ) -> PINNOutput:
        ys = []
        for i, x in enumerate(x_list):
            ys.append(self.subnets[i].predict(x))

        # Interface continuity loss
        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=ys[0].device)}
        if interface_pairs:
            iface = torch.tensor(0.0, device=ys[0].device)
            for (i, j, xi, xj) in interface_pairs:
                yi = self.subnets[i].predict(xi)
                yj = self.subnets[j].predict(xj)
                iface = iface + torch.mean((yi - yj) ** 2)
            losses["interface"] = iface
            losses["total"] = losses["total"] + self.interface_weight * iface

        # Optional physics loss per subnet
        if physics_fn is not None and physics_data_list is not None:
            phys_total = torch.tensor(0.0, device=ys[0].device)
            for i, pdata in enumerate(physics_data_list):
                pl = self.subnets[i].physics_loss(physics_fn=physics_fn, physics_data=pdata)
                phys_total = phys_total + pl.get("physics", torch.tensor(0.0, device=ys[0].device))
            losses["physics"] = phys_total
            losses["total"] = losses["total"] + phys_total

        # output y is a list (XPINN is multi-domain)
        return PINNOutput(y=ys, losses=losses, extras={"n_subdomains": len(self.subnets)})
