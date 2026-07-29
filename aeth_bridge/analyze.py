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


def _nearest_distances(
    source: np.ndarray, target: np.ndarray, chunk: int = 512
) -> np.ndarray:
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
    return sorted(
        clusters, key=lambda item: item["areaFraction"], reverse=True
    )[:limit]


def _planar_regions(
    mesh: trimesh.Trimesh, diagonal: float, limit: int = 16
) -> list[dict[str, Any]]:
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if len(normals) == 0:
        return []
    scale = max(diagonal, 1e-12)
    normal_keys = np.round(normals, decimals=2)
    offsets = np.einsum("ij,ij->i", centers, normals)
    offset_keys = np.round(offsets / scale, decimals=2)
    keys = np.column_stack([normal_keys, offset_keys])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    total_area = max(float(areas.sum()), 1e-12)
    regions: list[dict[str, Any]] = []
    for index, key in enumerate(unique):
        mask = inverse == index
        area = float(areas[mask].sum())
        if area / total_area < 0.01:
            continue
        weighted_normal = np.average(normals[mask], axis=0, weights=areas[mask])
        length = float(np.linalg.norm(weighted_normal))
        if length <= 1e-12:
            continue
        weighted_normal /= length
        region_offsets = centers[mask] @ weighted_normal
        offset = float(np.average(region_offsets, weights=areas[mask]))
        residual = float(np.sqrt(np.average((region_offsets - offset) ** 2, weights=areas[mask])))
        regions.append(
            {
                "normal": _vector(weighted_normal),
                "offset": offset,
                "areaFraction": area / total_area,
                "faceCount": int(mask.sum()),
                "normalizedResidual": residual / scale,
                "confidence": float(np.exp(-100.0 * residual / scale)),
            }
        )
    return sorted(regions, key=lambda item: item["areaFraction"], reverse=True)[:limit]


def _cylinder_candidates(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    axes: np.ndarray,
    diagonal: float,
) -> list[dict[str, Any]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total_area = max(float(areas.sum()), 1e-12)
    candidates: list[dict[str, Any]] = []
    centered = vertices - origin
    for axis_index in range(3):
        axis = axes[:, axis_index]
        side_faces = np.abs(normals @ axis) < 0.25
        side_area_fraction = float(areas[side_faces].sum() / total_area)
        if side_area_fraction < 0.2 or not np.any(side_faces):
            continue
        indices = np.unique(faces[side_faces].reshape(-1))
        points = centered[indices]
        axial = points @ axis
        radial_vectors = points - np.outer(axial, axis)
        radii = np.linalg.norm(radial_vectors, axis=1)
        radius = float(np.median(radii))
        if radius <= max(diagonal, 1e-12) * 1e-4:
            continue
        residual = float(np.median(np.abs(radii - radius)) / radius)
        length = float(axial.max() - axial.min())
        confidence = float(np.exp(-18.0 * residual) * min(1.0, side_area_fraction * 2.0))
        if confidence < 0.2:
            continue
        candidates.append(
            {
                "axis": _vector(axis),
                "origin": _vector(origin),
                "radius": radius,
                "length": length,
                "normalizedResidual": residual,
                "supportAreaFraction": side_area_fraction,
                "confidence": confidence,
            }
        )
    return sorted(candidates, key=lambda item: item["confidence"], reverse=True)


def _sphere_candidate(
    vertices: np.ndarray, origin: np.ndarray, diagonal: float
) -> dict[str, Any] | None:
    radii = np.linalg.norm(vertices - origin, axis=1)
    radius = float(np.median(radii))
    if radius <= max(diagonal, 1e-12) * 1e-4:
        return None
    residual = float(np.median(np.abs(radii - radius)) / radius)
    confidence = float(np.exp(-20.0 * residual))
    if confidence < 0.2:
        return None
    return {
        "origin": _vector(origin),
        "radius": radius,
        "normalizedResidual": residual,
        "confidence": confidence,
    }


def _edge_summary(mesh: trimesh.Trimesh) -> dict[str, Any]:
    inverse = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
    incidence = np.bincount(inverse, minlength=len(mesh.edges_unique))
    angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    sharp = angles >= np.deg2rad(30.0)
    sharp_angles = np.rad2deg(angles[sharp])
    return {
        "boundaryEdgeCount": int(np.count_nonzero(incidence == 1)),
        "nonManifoldEdgeCount": int(np.count_nonzero(incidence > 2)),
        "sharpEdgeCount": int(np.count_nonzero(sharp)),
        "medianSharpAngleDegrees": (
            float(np.median(sharp_angles)) if len(sharp_angles) else None
        ),
    }


def _symmetry_scores(
    vertices: np.ndarray, center: np.ndarray, sample_limit: int = 4096
) -> list[dict[str, Any]]:
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


def _shape_class(
    planar: list[dict[str, Any]],
    cylinders: list[dict[str, Any]],
    sphere: dict[str, Any] | None,
) -> tuple[str, float]:
    planar_support = sum(item["areaFraction"] for item in planar[:6])
    if sphere is not None and sphere["confidence"] >= 0.7:
        return "spherical", float(sphere["confidence"])
    if cylinders and cylinders[0]["confidence"] >= 0.55:
        return "cylindrical", float(cylinders[0]["confidence"])
    if planar_support >= 0.75:
        return "prismatic", float(min(1.0, planar_support))
    return "freeform-or-mixed", float(max(0.1, 1.0 - planar_support))


def analyze_mesh(mesh: trimesh.Trimesh, *, source: str | None = None) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    center = bounds.mean(axis=0)
    diagonal = float(np.linalg.norm(extents))
    eigenvalues, axes = _principal_frame(vertices)
    origin = vertices.mean(axis=0)
    components = list(mesh.split(only_watertight=False))
    planar = _planar_regions(mesh, diagonal)
    cylinders = _cylinder_candidates(mesh, origin, axes, diagonal)
    sphere = _sphere_candidate(vertices, origin, diagonal)
    shape_class, shape_confidence = _shape_class(planar, cylinders, sphere)
    euler_number = int(mesh.euler_number)
    component_count = max(len(components), 1)
    genus = (
        max(0, int(round((2 * component_count - euler_number) / 2)))
        if mesh.is_watertight
        else None
    )
    sorted_extents = np.sort(np.maximum(extents, 0.0))
    largest_extent = max(float(sorted_extents[-1]), 1e-12)
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
            "variances": _vector(eigenvalues),
        },
        "normalClusters": _normal_clusters(mesh),
        "analyticCandidates": {
            "planes": planar,
            "cylinders": cylinders,
            "spheres": [] if sphere is None else [sphere],
        },
        "edgeSummary": _edge_summary(mesh),
        "shapeDescriptor": {
            "classification": shape_class,
            "confidence": shape_confidence,
            "eulerNumber": euler_number,
            "genus": genus,
            "thinnessRatios": [
                float(sorted_extents[0] / largest_extent),
                float(sorted_extents[1] / largest_extent),
            ],
        },
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
            for component in sorted(
                components, key=lambda item: item.area, reverse=True
            )[:64]
        ],
        "uncertainty": {
            "scaleKnown": False,
            "requiresScaleAnchor": True,
            "recommendedObservation": "Provide one real-world dimension or a calibrated orthographic view.",
        },
    }


def analyze_path(path: str) -> dict[str, Any]:
    return analyze_mesh(load_mesh(path), source=path)
