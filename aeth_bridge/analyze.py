from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from .meshio import load_mesh

REPORT_VERSION = 1


def _vector(value: np.ndarray) -> list[float]:
    return [float(component) for component in value.tolist()]


def _principal_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = vertices - vertices.mean(axis=0)
    covariance = centered.T @ centered / max(len(vertices), 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    if np.linalg.det(vectors) < 0:
        vectors[:, -1] *= -1
    return values, vectors


def _nearest_distances(source: np.ndarray, target: np.ndarray, chunk: int = 512) -> np.ndarray:
    result = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), chunk):
        part = source[start : start + chunk]
        squared = np.sum((part[:, None, :] - target[None, :, :]) ** 2, axis=2)
        result[start : start + len(part)] = np.sqrt(np.min(squared, axis=1))
    return result


def _normal_clusters(mesh: trimesh.Trimesh, limit: int = 12) -> list[dict[str, Any]]:
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if len(normals) == 0:
        return []
    rounded = np.round(normals, decimals=2)
    keys, inverse = np.unique(rounded, axis=0, return_inverse=True)
    total_area = max(float(areas.sum()), 1e-12)
    clusters = []
    for index, key in enumerate(keys):
        mask = inverse == index
        weight = float(areas[mask].sum())
        clusters.append(
            {
                "normal": _vector(key),
                "areaFraction": weight / total_area,
                "faceCount": int(mask.sum()),
            }
        )
    return sorted(clusters, key=lambda item: item["areaFraction"], reverse=True)[:limit]


def _symmetry_scores(vertices: np.ndarray, center: np.ndarray, sample_limit: int = 4096) -> list[dict[str, Any]]:
    if len(vertices) > sample_limit:
        indices = np.linspace(0, len(vertices) - 1, sample_limit, dtype=np.int64)
        points = vertices[indices]
    else:
        points = vertices
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    scale = max(diagonal, 1e-12)
    output = []
    for axis, name in enumerate(("YZ", "XZ", "XY")):
        mirrored = points.copy()
        mirrored[:, axis] = 2.0 * center[axis] - mirrored[:, axis]
        distances = _nearest_distances(mirrored, points, chunk=256)
        normalized = float(np.median(distances) / scale)
        output.append(
            {
                "plane": name,
                "normalizedMedianResidual": normalized,
                "confidence": float(np.exp(-40.0 * normalized)),
            }
        )
    return sorted(output, key=lambda item: item["confidence"], reverse=True)


def analyze_mesh(mesh: trimesh.Trimesh, *, source: str | None = None) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    center = bounds.mean(axis=0)
    eigenvalues, axes = _principal_frame(vertices)
    components = list(mesh.split(only_watertight=False))
    return {
        "version": REPORT_VERSION,
        "source": source,
        "mesh": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "components": int(len(components)),
            "watertight": bool(mesh.is_watertight),
            "windingConsistent": bool(mesh.is_winding_consistent),
            "surfaceArea": float(mesh.area),
            "signedVolume": float(mesh.volume) if mesh.is_volume else None,
        },
        "bounds": {
            "minimum": _vector(bounds[0]),
            "maximum": _vector(bounds[1]),
            "center": _vector(center),
            "extents": _vector(extents),
            "diagonal": float(np.linalg.norm(extents)),
        },
        "principalFrame": {
            "origin": _vector(vertices.mean(axis=0)),
            "axes": [_vector(axes[:, index]) for index in range(3)],
            "variances": _vector(eigenvalues),
        },
        "normalClusters": _normal_clusters(mesh),
        "symmetryCandidates": _symmetry_scores(vertices, center),
        "componentSummaries": [
            {
                "vertices": int(len(component.vertices)),
                "faces": int(len(component.faces)),
                "surfaceArea": float(component.area),
                "watertight": bool(component.is_watertight),
                "bounds": {
                    "minimum": _vector(np.asarray(component.bounds)[0]),
                    "maximum": _vector(np.asarray(component.bounds)[1]),
                },
            }
            for component in sorted(components, key=lambda item: item.area, reverse=True)[:64]
        ],
        "uncertainty": {
            "scaleKnown": False,
            "requiresScaleAnchor": True,
            "recommendedObservation": "Provide one real-world dimension or a calibrated view.",
        },
    }


def analyze_path(path: str) -> dict[str, Any]:
    return analyze_mesh(load_mesh(path), source=path)
