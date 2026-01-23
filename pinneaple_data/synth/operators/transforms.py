import torch

def normalize_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mn = x.amin()
    mx = x.amax()
    return (x - mn) / (mx - mn).clamp_min(eps)
