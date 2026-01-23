"""
Parametric variants MVP: generate CAD -> trimesh -> MeshData -> sample surface points.

Requires:
  - cadquery (OCC stack)
  - trimesh
"""

from __future__ import annotations

try:
    import cadquery as cq
except Exception as e:
    print("cadquery not available in this environment:", e)
    raise SystemExit(0)

import numpy as np

from pinneaple_geom.gen.cadquery_gen import cadquery_to_trimesh
from pinneaple_geom.core.mesh import MeshData
from pinneaple_geom.sample.points import sample_surface_points


def _tm_to_meshdata(tm) -> MeshData:
    import numpy as np
    return MeshData(
        vertices=tm.vertices.view(np.ndarray),
        faces=tm.faces.view(np.ndarray),
        normals=tm.face_normals.view(np.ndarray) if getattr(tm, "face_normals", None) is not None else None,
    )


def make_box_cq(width: float, depth: float, height: float):
    """
    Create a parametric box using CadQuery.
    Returns a CadQuery Workplane / Shape.
    """
    return cq.Workplane("XY").box(width, depth, height, centered=(True, True, True))


def main():
    # Base primitive (parametric)
    base = make_box_cq(width=1.0, depth=1.0, height=0.25)

    # Variants (simple param sweep)
    variants = []
    for w in [0.8, 1.0, 1.2]:
        for h in [0.2, 0.3]:
            shape = make_box_cq(width=w, depth=1.0, height=h)
            variants.append((w, h, shape))

    for (w, h, shape) in variants[:5]:
        tm = cadquery_to_trimesh(shape)  # trimesh.Trimesh

        # Convert to MeshData for a stable internal API
        mesh = _tm_to_meshdata(tm)

        # sample points (MeshData path)
        pts, normals, face_id = sample_surface_points(mesh, n=2000)
        pts = np.asarray(pts)
        
        print(f"variant w={w} h={h} | faces={len(mesh.faces)} | pts={pts.shape}")


if __name__ == "__main__":
    main()
