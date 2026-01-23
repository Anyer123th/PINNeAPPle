from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class HamiltonianNeuralNetwork(ContinuousModelBase):
    """
    HNN MVP:
      - Learn Hamiltonian H(q,p)
      - Dynamics: dq/dt = dH/dp, dp/dt = -dH/dq
    """
    def __init__(self, dim_q: int, hidden: int = 128, num_layers: int = 3):
        super().__init__()
        self.dim_q = int(dim_q)
        self.dim_p = int(dim_q)

        layers = [nn.Linear(2 * dim_q, hidden), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.H = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,  # (B,2*dim_q) = [q,p]
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> ContOutput:
        z = z.requires_grad_(True)
        H = self.H(z).sum()
        grad = torch.autograd.grad(H, z, create_graph=True)[0]

        qdot = grad[:, self.dim_q:]
        pdot = -grad[:, :self.dim_q]
        dz = torch.cat([qdot, pdot], dim=-1)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=z.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(dz, y_true)
            losses["total"] = losses["mse"]

        return ContOutput(y=dz, losses=losses, extras={"H": H})
