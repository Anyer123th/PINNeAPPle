from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .geometry import GeometrySpec, GeometryAsset
from .mesh import MeshData


def _rotation_matrix_xyz(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]])
    Ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]])
    Rz = np.array([[cz,-sz,0,0],[sz,cz,0,0],[0,0,1,0],[0,0,0,1]])
    return Rz @ Ry @ Rx


def _load_trimesh_from_file(path: Path):
    import trimesh
    m = trimesh.load(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError("Loaded geometry is not a triangular mesh.")
    return m


def _meshdata_from_trimesh(tm) -> MeshData:
    return MeshData(
        vertices=tm.vertices.view(np.ndarray),
        faces=tm.faces.view(np.ndarray),
        normals=tm.face_normals.view(np.ndarray) if tm.face_normals is not None else None,
    )


def _meshdata_from_meshio(mesh) -> MeshData:
    if "triangle" not in mesh.cells_dict:
        raise ValueError("Mesh does not contain triangle cells.")
    faces = mesh.cells_dict["triangle"]
    vertices = mesh.points[:, :3]
    return MeshData(vertices=vertices, faces=faces)


def build_geometry_asset(
    spec: Dict[str, Any] | GeometrySpec,
    *,
    options: Optional[Any] = None,
) -> GeometryAsset:
    """
    Build a GeometryAsset from a GeometrySpec or spec dict.

    Supported kinds (MVP):
      - file       (STL/OBJ/PLY/GLTF) via trimesh
      - mesh_file  (VTK/VTU/MSH/...) via meshio
      - primitive  (delegated to pinneaple_geom.gen.primitives)
    """
    if isinstance(spec, dict):
        spec = GeometrySpec(**spec)

    kind = spec.kind.lower()

    # ---------- Load mesh ----------
    if kind in ("file", "mesh_file"):
        path = Path(spec.path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        if kind == "file":
            tm = _load_trimesh_from_file(path)
            mesh = _meshdata_from_trimesh(tm)

        else:  # mesh_file
            import meshio
            mio = meshio.read(path)
            mesh = _meshdata_from_meshio(mio)

    elif kind == "primitive":
        try:
            from pinneaple_geom.gen.primitives import build_primitive  # type: ignore
        except Exception as e:
            raise ImportError("Primitive generation requires pinneaple_geom.gen.primitives") from e
        mesh = build_primitive(spec.name, **spec.params)

    else:
        raise ValueError(f"Unsupported GeometrySpec kind: {kind}")

    # ---------- Apply options ----------
    if options:
        if getattr(options, "scale", None):
            mesh.vertices *= float(options.scale)

        if getattr(options, "rotate_euler_deg", None):
            rx, ry, rz = options.rotate_euler_deg
            R = _rotation_matrix_xyz(
                np.deg2rad(rx),
                np.deg2rad(ry),
                np.deg2rad(rz),
            )
            mesh.apply_transform(R)

        if getattr(options, "translate", None):
            t = np.array(options.translate, dtype=np.float64)
            mesh.vertices += t[None, :]

    # ---------- Build asset ----------
    bmin, bmax = mesh.bounds()
    asset = GeometryAsset(
        mesh=mesh,
        bounds=(bmin, bmax),
        units=getattr(options, "units", None),
        boundary_groups=getattr(options, "boundary_labels", {}) or {},
        meta={
            "kind": kind,
            "source": spec.path or spec.name,
        },
    )
    return asset


def load_geometry_asset(
    geom: Any,
    *,
    options: Optional[Any] = None,
) -> GeometryAsset:
    """
    Convenience wrapper:
      - Path / str -> infer spec
      - GeometryAsset -> returned as-is
      - dict -> GeometrySpec
    """
    if isinstance(geom, GeometryAsset):
        return geom

    if isinstance(geom, (str, Path)):
        p = Path(geom)
        ext = p.suffix.lower().lstrip(".")
        kind = "file" if ext in ("stl","obj","ply","glb","gltf","off") else "mesh_file"
        spec = GeometrySpec(kind=kind, path=str(p))
        return build_geometry_asset(spec, options=options)

    if isinstance(geom, dict):
        return build_geometry_asset(geom, options=options)

    raise TypeError(f"Unsupported geometry input type: {type(geom)}")
