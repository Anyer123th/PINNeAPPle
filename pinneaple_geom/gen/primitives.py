from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from pinneaple_geom.core.mesh import MeshData


def _tm_to_meshdata(tm) -> MeshData:
    # trimesh.Trimesh -> MeshData
    return MeshData(
        vertices=tm.vertices.view(np.ndarray),
        faces=tm.faces.view(np.ndarray),
        normals=tm.face_normals.view(np.ndarray) if getattr(tm, "face_normals", None) is not None else None,
    )


def _as_3tuple(v: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        x = float(v)
        return (x, x, x)
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    raise TypeError(f"Expected a scalar or 3-tuple, got: {type(v).__name__} {v}")


def build_primitive(name: str, **params) -> MeshData:
    """
    Build simple parametric primitives (tri meshes) using trimesh.

    Supported primitives (MVP):
      - "box":       extents=(x,y,z) or size=(x,y,z) or side=s
                    aliases: "cube", "rect", "cuboid"
      - "sphere":    radius=..., subdivisions=...
      - "cylinder":  radius=..., height=..., sections=...
      - "plane":     size=(x,y) (z=0) as two triangles
      - "channel":   length, width, height  (duct as box for MVP)

    Notes:
      - Intended for sampling and simple domains.
      - More complex CAD belongs in cadquery_gen.
    """
    import trimesh

    key = (name or "").lower().strip()

    # ---- BOX / CUBE
    if key in {"box", "cube", "rect", "cuboid"}:
        # allow: extents, size, (x,y,z), or side
        if "side" in params and (params.get("extents") is None and params.get("size") is None):
            extents = _as_3tuple(params.get("side"), (1.0, 1.0, 1.0))
        else:
            extents = _as_3tuple(params.get("extents") or params.get("size"), (1.0, 1.0, 1.0))

        # guard against zeros/negatives (trimesh may behave weirdly)
        ex = tuple(max(1e-12, float(e)) for e in extents)
        tm = trimesh.creation.box(extents=ex)
        return _tm_to_meshdata(tm)

    # ---- SPHERE
    if key == "sphere":
        radius = float(params.get("radius", 1.0))
        subdivisions = int(params.get("subdivisions", 3))
        tm = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        return _tm_to_meshdata(tm)

    # ---- CYLINDER
    if key == "cylinder":
        radius = float(params.get("radius", 1.0))
        height = float(params.get("height", 1.0))
        sections = int(params.get("sections", 32))
        tm = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
        return _tm_to_meshdata(tm)

    # ---- PLANE
    if key == "plane":
        size = params.get("size") or (1.0, 1.0)
        sx, sy = float(size[0]), float(size[1])

        v = np.array(
            [
                [-sx / 2, -sy / 2, 0.0],
                [ sx / 2, -sy / 2, 0.0],
                [ sx / 2,  sy / 2, 0.0],
                [-sx / 2,  sy / 2, 0.0],
            ],
            dtype=np.float64,
        )
        f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return MeshData(vertices=v, faces=f, normals=n)

    # ---- CHANNEL (duct as box MVP)
    if key == "channel":
        length = float(params.get("length", 1.0))
        width = float(params.get("width", 0.2))
        height = float(params.get("height", 0.2))
        tm = trimesh.creation.box(extents=(length, width, height))
        return _tm_to_meshdata(tm)

    raise ValueError(f"Unsupported primitive '{name}'. Supported: box/cube, sphere, cylinder, plane, channel.")
