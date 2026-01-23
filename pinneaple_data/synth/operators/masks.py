import torch

def random_block_mask(H: int, W: int, frac: float = 0.2, seed: int = 42) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    m = torch.zeros((H, W), dtype=torch.bool)
    h = max(1, int(H * frac))
    w = max(1, int(W * frac))
    i = int(torch.randint(0, max(1, H - h), (1,), generator=g).item())
    j = int(torch.randint(0, max(1, W - w), (1,), generator=g).item())
    m[i:i + h, j:j + w] = True
    return m
