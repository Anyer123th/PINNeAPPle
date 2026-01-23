from __future__ import annotations

from typing import Any, Optional, Callable
import torch
import torch.nn as nn

from .base import SynthConfig, SynthOutput
from .pde import SimplePhysicalSample


def _as_tensor(img: Any, device, dtype) -> torch.Tensor:
    if isinstance(img, torch.Tensor):
        t = img
    else:
        t = torch.tensor(img)
    return t.to(device=device, dtype=dtype)


def _smooth2d(x: torch.Tensor) -> torch.Tensor:
    """
    Simple 2D smoothing via neighbor averaging.
    x: (H,W) or (C,H,W)
    """
    if x.ndim == 2:
        x2 = x[None, None, :, :]  # (1,1,H,W)
    elif x.ndim == 3:
        x2 = x[None, :, :, :]     # (1,C,H,W)
    else:
        raise ValueError("Expected (H,W) or (C,H,W)")

    # 4-neighbor averaging kernel
    k = torch.tensor(
        [[0.0, 0.25, 0.0],
         [0.25, 0.0, 0.25],
         [0.0, 0.25, 0.0]],
        device=x.device, dtype=x.dtype
    )[None, None, :, :]  # (1,1,3,3)

    if x2.shape[1] > 1:
        k = k.repeat(x2.shape[1], 1, 1, 1)  # depthwise
        y = torch.nn.functional.conv2d(x2, k, padding=1, groups=x2.shape[1])
    else:
        y = torch.nn.functional.conv2d(x2, k, padding=1)

    y = y[0]
    return y[0] if y.shape[0] == 1 else y


class ImageReconstructionSynthGenerator:
    """
    Image reconstruction / synthesis generator.

    MVP modes:
      - "inpaint_smooth": iterative smoothing inside masked region
      - "autoencoder": use provided AE model to reconstruct

    Input:
      img: (H,W) or (C,H,W)
      mask: same spatial shape (H,W) bool or 0/1
        mask==1 means MISSING region to reconstruct
    """
    def __init__(self, cfg: Optional[SynthConfig] = None):
        self.cfg = cfg or SynthConfig()

    @torch.no_grad()
    def generate(
        self,
        *,
        img: Any,
        mask: Optional[Any] = None,
        mode: str = "inpaint_smooth",
        steps: int = 200,
        ae_model: Optional[nn.Module] = None,
        post_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> SynthOutput:
        device = torch.device(self.cfg.device)
        dtype = getattr(torch, self.cfg.dtype)

        x = _as_tensor(img, device, dtype)
        mode = mode.lower().strip()

        if mask is None:
            # default: random missing block
            H, W = (x.shape[-2], x.shape[-1]) if x.ndim == 3 else x.shape
            m = torch.zeros((H, W), device=device, dtype=torch.bool)
            h0 = int(0.3 * H)
            w0 = int(0.3 * W)
            m[h0:h0 + int(0.2 * H), w0:w0 + int(0.2 * W)] = True
        else:
            m = _as_tensor(mask, device, torch.float32) > 0.5
            if m.ndim == 3:
                m = m[0]  # assume (1,H,W) or (C,H,W) same mask for all

        # create corrupted
        x_cor = x.clone()
        if x.ndim == 2:
            x_cor[m] = 0.0
        else:
            x_cor[:, m] = 0.0

        if mode == "inpaint_smooth":
            rec = x_cor.clone()
            for _ in range(int(steps)):
                sm = _smooth2d(rec)
                if rec.ndim == 2:
                    rec[m] = sm[m]
                else:
                    rec[:, m] = sm[:, m]
        elif mode == "autoencoder":
            if ae_model is None:
                raise ValueError("ae_model is required for mode='autoencoder'")
            ae_model = ae_model.to(device).eval()
            inp = x_cor
            if inp.ndim == 2:
                inp = inp[None, None, :, :]
            elif inp.ndim == 3:
                inp = inp[None, :, :, :]
            rec = ae_model(inp)[0]
            # rec shape (C,H,W) or (1,H,W)
            if rec.shape[0] == 1 and x.ndim == 2:
                rec = rec[0]
        else:
            raise ValueError("mode must be inpaint_smooth | autoencoder")

        if post_fn is not None:
            rec = post_fn(rec)

        sample = SimplePhysicalSample(
            fields={"img": x, "img_corrupt": x_cor, "img_recon": rec, "mask": m.to(dtype=torch.float32)},
            coords={},
            meta={"mode": mode, "steps": int(steps)},
        )
        return SynthOutput(samples=[sample], extras={"shape": tuple(x.shape)})
