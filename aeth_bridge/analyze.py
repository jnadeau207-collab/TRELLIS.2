from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from .meshio import load_mesh

REPORT_VERSION = 1
_EPS = 1e-12


def _vector(value: np.ndarray) -> list[float]:
    return [float(component) for component in value.tolist()]


def _principal_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = vertices - vertices.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered / max(len(vertices), 1))
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0.0), vectors[:, order]
    if np.linalg.det(vectors) < 0:
        vectors[:, -1] *= -1
    return values, vectors


def _nearest(source: np.ndarray, target: np.ndarray, chunk: int = 256) -> np.ndarray:
    result = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), chunk):
        part = source[start : start + chunk]
        squared = np.sum((part[:, None, :] - target[None, :, :]) ** 2, axis=2)
        result[start : start + len(part)] = np.sqrt(np.min(squared, axis=1))
    return result


def _normal_clusters(mesh: trimesh.Trimesh) -> list[dict[str, Any]]:
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if not len(normals):
        return []
    keys, inverse = np.unique(np.round(normals, 2), axis=0, return_inverse=True)
    total = max(float(areas.sum()), _EPS)
    output = []
    for index, normal in enumerate(keys):
        mask = inverse == index
        output.append(
            {
                "normal": _vector(normal),
                "areaFraction": float(areas[mask].sum() / total),
                "faceCount": int(mask.sum()),
            }
        )
    return sorted(output, key=lambda item: item["areaFraction"], reverse=True)[:12]


def _planar_regions(mesh: trimesh.Trimesh, diagonal: float) -> list[dict[str, Any]]:
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if not len(normals):
        return []
    scale = max(diagonal, _EPS)
    offsets = np.einsum("ij,ij->i", centers, normals)
    keys = np.column_stack([np.round(normals, 2), np.round(offsets / scale, 2)])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    total = max(float(areas.sum()), _EPS)
    regions = []
    for index in range(len(unique)):
        mask = inverse == index
        area = float(areas[mask].sum())
        if area / total < 0.01:
            continue
        normal = np.average(normals[mask], axis=0, weights=areas[mask])
        normal /= max(float(np.linalg.norm(normal)), _EPS)
        local_offsets = centers[mask] @ normal
        offset = float(np.average(local_offsets, weights=areas[mask]))
        residual = float(
            np.sqrt(np.average((local_offsets - offset) ** 2, weights=areas[mask]))
        )
        regions.append(
            {
                "normal": _vector(normal),
                "offset": offset,
                "areaFraction": area / total,
                "faceCount": int(mask.sum()),
                "normalizedResidual": residual / scale,
                "confidence": float(np.exp(-100.0 * residual / scale)),
            }
        )
    return sorted(regions, key=lambda item: item["areaFraction"], reverse=True)[:16]


def _cylinders(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    axes: np.ndarray,
    diagonal: float,
) -> list[dict[str, Any]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total = max(float(areas.sum()), _EPS)
    centered = vertices - origin
    output = []
    for axis in axes.T:
        side_faces = np.abs(normals @ axis) < 0.25
        support = float(areas[side_faces].sum() / total)
        if support < 0.2 or not np.any(side_faces):
            continue
        points = centered[np.unique(faces[side_faces].reshape(-1))]
        axial = points @ axis
        radial = points - np.outer(axial, axis)
        radii = np.linalg.norm(radial, axis=1)
        radius = float(np.median(radii))
        if radius <= max(diagonal, _EPS) * 1e-4:
            continue
        residual = float(np.median(np.abs(radii - radius)) / radius)
        confidence = float(np.exp(-18.0 * residual) * min(1.0, support * 2.0))
        if confidence >= 0.2:
            output.append(
                {
                    "axis": _vector(axis),
                    "origin": _vector(origin),
                    "radius": radius,
                    "length": float(axial.max() - axial.min()),
                    "normalizedResidual": residual,
                    "supportAreaFraction": support,
                    "confidence": confidence,
                }
            )
    return sorted(output, key=lambda item: item["confidence"], reverse=True)[:3]


def _sphere(
    mesh: trimesh.Trimesh, origin: np.ndarray, diagonal: float
) -> dict[str, Any] | None:
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if not len(centers):
        return None
    radial = centers - origin
    radii = np.linalg.norm(radial, axis=1)
    radius = float(np.average(radii, weights=areas))
    if radius <= max(diagonal, _EPS) * 1e-4:
        return None
    residual = float(
        np.sqrt(np.average((radii - radius) ** 2, weights=areas)) / radius
    )
    valid = radii > _EPS
    if not np.any(valid):
        return None
    directions = radial[valid] / radii[valid, None]
    alignment = float(
        np.average(
            np.abs(np.einsum("ij,ij->i", directions, normals[valid])),
            weights=areas[valid],
        )
    )
    confidence = float(np.exp(-20.0 * residual) * alignment**4)
    if confidence < 0.2:
        return None
    return {
        "origin": _vector(origin),
        "radius": radius,
        "normalizedResidual": residual,
        "confidence": confidence,
    }


def _edge_summary(mesh: trimesh.Trimesh) -> dict[str, Any]:
    incidence = np.bincount(
        np.asarray(mesh.edges_unique_inverse, dtype=np.int64),
        minlength=len(mesh.edges_unique),
    )
    angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    sharp_angles = np.rad2deg(angles[angles >= np.deg2rad(30.0)])
    return {
        "boundaryEdgeCount": int(np.count_nonzero(incidence == 1)),
        "nonManifoldEdgeCount": int(np.count_nonzero(incidence > 2)),
        "sharpEdgeCount": int(len(sharp_angles)),
        "medianSharpAngleDegrees": (
            float(np.median(sharp_angles)) if len(sharp_angles) else None
        ),
    }


def _symmetry(vertices: np.ndarray, center: np.ndarray) -> list[dict[str, Any]]:
    points = (
        vertices[np.linspace(0, len(vertices) - 1, 4096, dtype=np.int64)]
        if len(vertices) > 4096
        else vertices
    )
    scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), _EPS)
    output = []
    for axis, plane in enumerate(("YZ", "XZ", "XY")):
        mirrored = points.copy()
        mirrored[:, axis] = 2.0 * center[axis] - mirrored[:, axis]
        normalized = float(np.median(_nearest(mirrored, points)) / scale)
        output.append(
            {
                "plane": plane,
                "normalizedMedianResidual": normalized,
                "confidence": float(np.exp(-40.0 * normalized)),
            }
        )
    return sorted(output, key=lambda item: item["confidence"], reverse=True)


def _shape_class(
    planes: list[dict[str, Any]],
    cylinders: list[dict[str, Any]],
    sphere: dict[str, Any] | None,
) -> tuple[str, float]:
    planar_support = sum(item["areaFraction"] for item in planes[:6])
    if planar_support >= 0.75:
        return "prismatic", float(min(1.0, planar_support))
    if cylinders and cylinders[0]["confidence"] >= 0.55:
        return "cylindrical", float(cylinders[0]["confidence"])
    if sphere is not None and sphere["confidence"] >= 0.7:
        return "spherical", float(sphere["confidence"])
    return "freeform-or-mixed", float(max(0.1, 1.0 - planar_support))


def analyze_mesh(mesh: trimesh.Trimesh, *, source: str | None = None) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents, center = bounds[1] - bounds[0], bounds.mean(axis=0)
    diagonal = float(np.linalg.norm(extents))
    values, axes = _principal_frame(vertices)
    origin = vertices.mean(axis=0)
    components = list(mesh.split(only_watertight=False))
    planes = _planar_regions(mesh, diagonal)
    cylinders = _cylinders(mesh, origin, axes, diagonal)
    sphere = _sphere(mesh, origin, diagonal)
    classification, confidence = _shape_class(planes, cylinders, sphere)
    euler = int(mesh.euler_number)
    component_count = max(len(components), 1)
    sorted_extents = np.sort(np.maximum(extents, 0.0))
    largest = max(float(sorted_extents[-1]), _EPS)
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
            "diagonal": diagonal,
        },
        "principalFrame": {
            "origin": _vector(origin),
            "axes": [_vector(axes[:, index]) for index in range(3)],
            "variances": _vector(values),
        },
        "normalClusters": _normal_clusters(mesh),
        "analyticCandidates": {
            "planes": planes,
            "cylinders": cylinders,
            "spheres": [] if sphere is None else [sphere],
        },
        "edgeSummary": _edge_summary(mesh),
        "shapeDescriptor": {
            "classification": classification,
            "confidence": confidence,
            "eulerNumber": euler,
            "genus": (
                max(0, int(round((2 * component_count - euler) / 2)))
                if mesh.is_watertight
                else None
            ),
            "thinnessRatios": [
                float(sorted_extents[0] / largest),
                float(sorted_extents[1] / largest),
            ],
        },
        "symmetryCandidates": _symmetry(vertices, center),
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
            "recommendedObservation": "Provide one real-world dimension or a calibrated orthographic view.",
        },
    }


def analyze_path(path: str) -> dict[str, Any]:
    return analyze_mesh(load_mesh(path), source=path)
