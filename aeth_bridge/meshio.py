from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh


class MeshError(ValueError):
    pass


def _as_mesh(value: Any) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        return value
    if isinstance(value, trimesh.Scene):
        geometries = [g for g in value.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise MeshError("scene contains no triangle mesh")
        return trimesh.util.concatenate(geometries)
    raise MeshError(f"unsupported mesh type: {type(value).__name__}")


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise MeshError(f"mesh file does not exist: {source}")
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            if "vertices" not in data or "faces" not in data:
                raise MeshError("NPZ mesh must contain vertices and faces")
            mesh = trimesh.Trimesh(
                vertices=np.asarray(data["vertices"], dtype=np.float64),
                faces=np.asarray(data["faces"], dtype=np.int64),
                process=False,
            )
    else:
        mesh = _as_mesh(trimesh.load(source, force=None, process=False))
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise MeshError("vertices must have shape [n, 3]")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise MeshError("faces must have shape [m, 3]")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise MeshError("mesh is empty")
    if not np.isfinite(mesh.vertices).all():
        raise MeshError("mesh contains non-finite vertices")
    return mesh


def save_npz(path: str | Path, vertices: Any, faces: Any) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    vertices_array = np.asarray(vertices, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.int32)
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise MeshError("vertices must have shape [n, 3]")
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise MeshError("faces must have shape [m, 3]")
    np.savez_compressed(target, vertices=vertices_array, faces=faces_array)
    return target
