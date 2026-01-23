from __future__ import annotations

from typing import Any, List, Optional, Tuple
import math
import numpy as np
import torch

from .base import SynthConfig, SynthOutput
from .pde import SimplePhysicalSample


class GeometrySynthGenerator:
    """
    Geometry synthesis/augmentation from an existing mesh (e.g., STL).

    MVP:
      - expects a trimesh.Trimesh-like object OR dict with vertices/faces
      - generates N variants preserving similarity

    Operations (controlled by params):
      - uniform scale in [smin, smax]
      - axis scale (sx,sy,sz) ranges
      - small random rotation
      - vertex noise (smooth-ish) with amplitude
    """
    def __init__(self, cfg: Optional[SynthConfig] = None):
        self.cfg = cfg or SynthConfig()

    def _rng(self):
        return np.random.default_rng(int(self.cfg.seed))

    def _extract(self, mesh: Any):
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            V = np.asarray(mesh.vertices)
            F = np.asarray(mesh.faces)
            return V, F
        if isinstance(mesh, dict) and "vertices" in mesh and "faces" in mesh:
            return np.asarray(mesh["vertices"]), np.asarray(mesh["faces"])
        raise TypeError("mesh must be trimesh-like or dict(vertices,faces)")

    def _rot_matrix(self, rx, ry, rz):
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return Rz @ Ry @ Rx

    def generate(
        self,
        *,
        mesh: Any,
        n_variants: int = 16,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        axis_scale_range: Tuple[float, float] = (0.95, 1.05),
        rot_deg: float = 5.0,
        noise_amp: float = 0.002,
        keep_centered: bool = True,
    ) -> SynthOutput:
        rng = self._rng()
        V0, F0 = self._extract(mesh)

        # center
        center = V0.mean(axis=0, keepdims=True)
        Vc = V0 - center if keep_centered else V0.copy()

        samples: List[SimplePhysicalSample] = []

        for i in range(int(n_variants)):
            V = Vc.copy()

            # scale
            s = rng.uniform(scale_range[0], scale_range[1])
            axs = rng.uniform(axis_scale_range[0], axis_scale_range[1], size=(3,))
            V = V * (s * axs)

            # rotation
            r = math.radians(rot_deg)
            rx, ry, rz = rng.uniform(-r, r), rng.uniform(-r, r), rng.uniform(-r, r)
            R = self._rot_matrix(rx, ry, rz)
            V = (V @ R.T)

            # vertex noise (MVP)
            if noise_amp and noise_amp > 0:
                n = rng.normal(0.0, noise_amp, size=V.shape)
                V = V + n

            # back center
            if keep_centered:
                V = V + center

            # store as tensors
            vt = torch.tensor(V, dtype=torch.float32)
            ft = torch.tensor(F0, dtype=torch.long)

            samples.append(
                SimplePhysicalSample(
                    fields={"vertices": vt, "faces": ft},
                    coords={},
                    meta={
                        "scale": float(s),
                        "axis_scale": axs.tolist(),
                        "rot_deg": float(rot_deg),
                        "noise_amp": float(noise_amp),
                    },
                )
            )

        return SynthOutput(samples=samples, extras={"n_variants": len(samples)})
