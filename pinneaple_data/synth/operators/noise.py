import torch

def add_gaussian_noise(x: torch.Tensor, std: float = 0.01, seed: int = 42) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return x + std * torch.randn_like(x, generator=g)
