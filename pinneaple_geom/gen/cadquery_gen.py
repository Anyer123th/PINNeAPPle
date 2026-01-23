from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from pinneaple_geom.core.mesh import MeshData


def cadquery_available() -> bool:
    try:
        import cadquery as cq  # noqa: F401
        return True
    except Exception:
        return False


def _tm_to_meshdata(tm) -> MeshData:
    return MeshData(
        vertices=tm.vertices.view(np.ndarray),
        faces=tm.faces.view(np.ndarray),
        normals=tm.face_normals.view(np.ndarray) if getattr(tm, "face_normals", None) is not None else None,
    )


@dataclass
class CadQueryTessellationOptions:
    """
    CadQuery tessellation controls (affect mesh fidelity and speed).

    Smaller deflection => more triangles (slower, higher fidelity)
    """
    linear_deflection: float = 0.1
    angular_deflection: float = 0.5


def cadquery_to_trimesh(
    cq_obj: Any,
    *,
    options: Optional[CadQueryTessellationOptions] = None,
):
    """
    Convert an in-memory CadQuery object (Workplane/Solid/Compound) to trimesh.Trimesh.

    MVP approach:
      - export to a temporary STL
      - load with trimesh (robust and avoids binding to CQ internals)

    Notes:
      - CadQuery export option support varies by installation.
      - 'options' is kept for API stability; advanced tessellation can be added later.
    """
    if not cadquery_available():
        raise ImportError("cadquery is not installed.")

    import tempfile
    import trimesh
    import cadquery as cq  # noqa: F401

    options = options or CadQueryTessellationOptions()

    with tempfile.TemporaryDirectory() as td:
        stl_path = Path(td) / "tmp.stl"

        # Many CadQuery environments ignore these options; keep simple for MVP.
        # If you later add tighter tessellation:
        # cq.exporters.export(cq_obj, str(stl_path), exportType="STL", tolerance=options.linear_deflection)
        cq.exporters.export(cq_obj, str(stl_path))

        tm = trimesh.load(str(stl_path), force="mesh")
        if not isinstance(tm, trimesh.Trimesh):
            raise TypeError("CadQuery export did not produce a triangular mesh (Trimesh).")
        return tm


def build_mesh_from_step(
    step_path: Union[str, Path],
    *,
    options: Optional[CadQueryTessellationOptions] = None,
) -> MeshData:
    """
    Load a STEP file with CadQuery and return a triangle mesh as MeshData.

    Requires:
      - cadquery installed
    """
    if not cadquery_available():
        raise ImportError("cadquery is not installed. Install it to use STEP/CAD import.")

    import cadquery as cq

    options = options or CadQueryTessellationOptions()

    step_path = Path(step_path).expanduser().resolve()
    if not step_path.exists():
        raise FileNotFoundError(step_path)

    cq_obj = cq.importers.importStep(str(step_path))
    tm = cadquery_to_trimesh(cq_obj, options=options)
    return _tm_to_meshdata(tm)


def build_mesh_from_cadquery_object(
    cq_obj: Any,
    *,
    options: Optional[CadQueryTessellationOptions] = None,
) -> MeshData:
    """
    Convert an in-memory CadQuery object (Workplane/Solid/Compound) to MeshData.

    Requires:
      - cadquery installed
    """
    tm = cadquery_to_trimesh(cq_obj, options=options)
    return _tm_to_meshdata(tm)
