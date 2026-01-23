from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class GeometrySpec:
    """
    Declarative geometry specification.

    Examples:
      {"kind":"primitive","name":"box","params":{...}}
      {"kind":"file","path":"model.stl"}
      {"kind":"mesh_file","path":"case.vtu"}
    """
    kind: str
    name: Optional[str] = None
    path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryAsset:
    """
    Unified geometry container used across Pinneaple.

    Holds a MeshData + metadata and optional boundary groups.

    Attributes:
      - mesh: MeshData (vertices, faces, normals)
      - bounds: (min_xyz, max_xyz)
      - units: optional physical units (e.g. meters)
      - boundary_groups: semantic labels (inlet/outlet/wall/etc.)
      - meta: free metadata (source, transforms applied, hashes)
    """
    mesh: Any  # MeshData
    bounds: Tuple[np.ndarray, np.ndarray]
    units: Optional[str] = None
    boundary_groups: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def bbox_size(self) -> np.ndarray:
        return self.bounds[1] - self.bounds[0]

    def center(self) -> np.ndarray:
        return 0.5 * (self.bounds[0] + self.bounds[1])

    def summary(self) -> Dict[str, Any]:
        return {
            "n_vertices": int(self.mesh.vertices.shape[0]),
            "n_faces": int(self.mesh.faces.shape[0]),
            "bounds": (self.bounds[0].tolist(), self.bounds[1].tolist()),
            "units": self.units,
            "boundary_groups": list(self.boundary_groups.keys()),
            "meta_keys": list(self.meta.keys()),
        }
