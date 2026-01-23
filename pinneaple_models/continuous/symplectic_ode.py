from __future__ import annotations
import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class SymplecticODENet(ContinuousModelBase):
    """
    Symplectic ODE-Net MVP:
      - Learns separable Hamiltonian: H(q,p)=T(p)+V(q)
      - Produces symplectic dynamics by construction.
    """
    def __init__(self, dim_q: int, hidden: int = 128):
        super().__init__()
        self.dim_q = int(dim_q)
        self.T = nn.Sequential(nn.Linear(dim_q, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.V = nn.Sequential(nn.Linear(dim_q, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor, *, y_true=None, return_loss=False) -> ContOutput:
        z = z.requires_grad_(True)
        q, p = z[:, :self.dim_q], z[:, self.dim_q:]

        H = (self.T(p) + self.V(q)).sum()
        grad = torch.autograd.grad(H, z, create_graph=True)[0]

        qdot = grad[:, self.dim_q:]
        pdot = -grad[:, :self.dim_q]
        dz = torch.cat([qdot, pdot], dim=-1)

        losses = {"total": torch.tensor(0.0, device=z.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(dz, y_true)
            losses["total"] = losses["mse"]

        return ContOutput(y=dz, losses=losses, extras={"H": H})
