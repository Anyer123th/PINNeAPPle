from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class MeshData:
    """
    Lightweight mesh container (numpy-only).

    vertices: (N,3) float64
    faces:    (M,3) int64 (triangle mesh)
    normals:  (M,3) or (N,3), optional
    """
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if self.normals is not None:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float64)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def apply_transform(self, matrix: np.ndarray) -> None:
        """
        Apply 4x4 homogeneous transform in-place.
        """
        if matrix.shape != (4, 4):
            raise ValueError("Transform matrix must be 4x4")
        v = np.ones((self.vertices.shape[0], 4), dtype=np.float64)
        v[:, :3] = self.vertices
        vt = (matrix @ v.T).T
        self.vertices = vt[:, :3]

    def copy(self) -> "MeshData":
        return MeshData(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=None if self.normals is None else self.normals.copy(),
        )
