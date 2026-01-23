from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, overload


# Accepts:
#  - path to geometry file (.stl, .obj, .ply, ...)
#  - path to mesh file (.vtu, .vtk, .msh, ...)
#  - dict spec {"kind": "...", ...}
#  - already-built GeometryAsset (from pinneaple_geom)
GeometryInput = Union[str, Path, Dict[str, Any], Any]


@dataclass
class GeometryLoadOptions:
    """
    Options for geometry/mesh loading and basic processing.

    This MVP focuses on:
      - fast load (trimesh/meshio)
      - basic transforms
      - optional simplification hooks (for later)

    Note: operations such as remeshing/decimation are expected to live in pinneaple_geom.ops.
    """
    # transforms
    scale: Optional[float] = None
    translate: Optional[Tuple[float, float, float]] = None
    rotate_euler_deg: Optional[Tuple[float, float, float]] = None  # (rx, ry, rz) degrees

    # optional processing flags
    repair: bool = True
    compute_normals: bool = True

    # optional labeling (boundary groups)
    # e.g., {"inlet": [...], "wall": [...]} (format depends on geom module)
    boundary_labels: Optional[Dict[str, Any]] = None


def _is_pathlike(x: Any) -> bool:
    try:
        Path(x)
        return isinstance(x, (str, Path))
    except Exception:
        return False


def _default_spec_from_path(p: Union[str, Path]) -> Dict[str, Any]:
    p = Path(p)
    ext = p.suffix.lower().lstrip(".")
    
    if ext in ("stl", "obj", "ply", "glb", "gltf", "off"):
        return {"kind": "file", "path": str(p)}
    if ext in ("vtk", "vtu", "msh", "mesh", "xdmf", "xmf"):
        return {"kind": "mesh_file", "path": str(p)}
    # fallback
    return {"kind": "file", "path": str(p)}


def load_geometry_asset(
    geom: GeometryInput,
    *,
    options: Optional[GeometryLoadOptions] = None,
) -> Any:
    """
    Loads/creates a GeometryAsset using pinneaple_geom.

    Returns:
      GeometryAsset (opaque here; defined in pinneaple_geom)

    Supported inputs:
      - Path/str: inferred spec by file extension
      - dict: GeometrySpec-like
      - GeometryAsset: returned as-is
    """
    options = options or GeometryLoadOptions()

    # If user passes an already-built asset, return it.
    # We avoid importing pinneaple_geom types explicitly (keeps adapters light).
    if not isinstance(geom, (str, Path, dict)):
        # Heuristic: treat as GeometryAsset if it has 'mesh' or 'vertices/faces'
        if hasattr(geom, "mesh") or hasattr(geom, "vertices") or hasattr(geom, "faces"):
            return geom

    if _is_pathlike(geom):
        spec = _default_spec_from_path(Path(geom))
    elif isinstance(geom, dict):
        spec = dict(geom)
    else:
        raise TypeError(f"Unsupported geometry input type: {type(geom)}")

    try:
        from pinneaple_geom.core.registry import build_geometry_asset  # type: ignore
    except Exception as e:
        raise ImportError(
            "pinneaple_geom is required to load geometry assets. "
            "Install geometry extras or add pinneaple_geom module."
        ) from e

    asset = build_geometry_asset(spec, options=options)  # type: ignore
    return asset


def attach_geometry(sample: Any, geom_asset: Any) -> Any:
    """
    Attaches a GeometryAsset to a PhysicalSample-like object.

    Expects `sample` to have `.geometry` and `.domain` (dict) or be dict-like.

    This keeps things flexible while we iterate on the PhysicalSample dataclass.
    """
    # dict-like support
    if isinstance(sample, dict):
        sample["geometry"] = geom_asset
        dom = sample.get("domain", {}) or {}
        dom.setdefault("type", "mesh")
        sample["domain"] = dom
        return sample

    # object-like support
    if hasattr(sample, "geometry"):
        setattr(sample, "geometry", geom_asset)
    if hasattr(sample, "domain"):
        dom = getattr(sample, "domain") or {}
        if not isinstance(dom, dict):
            dom = {"value": dom}
        dom.setdefault("type", "mesh")
        setattr(sample, "domain", dom)
    return sample
