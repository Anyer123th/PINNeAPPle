from __future__ import annotations

from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class BayesianRNN(ContinuousModelBase):
    """
    Bayesian RNN (MVP via MC Dropout / variational dropout approximation):

    - Train normally with dropout enabled.
    - At inference, do multiple stochastic forward passes to estimate uncertainty.

    Output:
      mu: (B,T,out_dim)
      extras:
        - "logvar": predictive log-variance (B,T,out_dim) estimated from MC samples
        - "samples": optionally returned if return_samples=True
    """
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        cell: Literal["gru", "lstm"] = "gru",
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        cell = (cell or "gru").lower().strip()
        if cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")
        self.cell = cell

        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def _single_pass(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        y = self.head(h)
        return y

    @torch.no_grad()
    def predict_mc(
        self,
        x: torch.Tensor,
        *,
        mc_samples: int = 16,
        return_samples: bool = False,
    ) -> ContOutput:
        # keep dropout ON for MC
        was_training = self.training
        self.train(True)

        ys = []
        for _ in range(int(mc_samples)):
            ys.append(self._single_pass(x))
        Y = torch.stack(ys, dim=0)  # (S,B,T,out_dim)

        mu = Y.mean(dim=0)
        var = Y.var(dim=0, unbiased=False).clamp_min(1e-12)
        logvar = torch.clamp(torch.log(var), self.min_logvar, self.max_logvar)

        # restore mode
        self.train(was_training)

        extras = {"logvar": logvar}
        if return_samples:
            extras["samples"] = Y

        return ContOutput(y=mu, losses={"total": torch.tensor(0.0, device=x.device)}, extras=extras)

    def forward(
        self,
        x: torch.Tensor,  # (B,T,input_dim)
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,T,out_dim)
        return_loss: bool = False,
        mc_samples: int = 0,   # if >0, will do MC in forward (slower)
    ) -> ContOutput:
        if mc_samples and mc_samples > 0:
            return self.predict_mc(x, mc_samples=mc_samples, return_samples=False)

        y_hat = self._single_pass(x)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y_hat, y_true)
            losses["total"] = losses["mse"]

        return ContOutput(y=y_hat, losses=losses, extras={})
