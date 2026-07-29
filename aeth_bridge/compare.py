from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from .meshio import load_mesh

COMPARISON_VERSION = 1


def _sample(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    count = max(128, min(int(count), 50_000))
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float64)


def _nearest(source: np.ndarray, target: np.ndarray, chunk: int = 256) -> np.ndarray:
    distances = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), chunk):
        part = source[start : start + chunk]
        squared = np.sum((part[:, None, :] - target[None, :, :]) ** 2, axis=2)
        distances[start : start + len(part)] = np.sqrt(np.min(squared, axis=1))
    return distances


def _relative_error(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    denominator = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denominator


def _alignment(
    reference_bounds: np.ndarray, candidate_bounds: np.ndarray
) -> tuple[float, np.ndarray]:
    reference_center = reference_bounds.mean(axis=0)
    candidate_center = candidate_bounds.mean(axis=0)
    reference_diagonal = float(
        np.linalg.norm(reference_bounds[1] - reference_bounds[0])
    )
    candidate_diagonal = float(
        np.linalg.norm(candidate_bounds[1] - candidate_bounds[0])
    )
    scale = reference_diagonal / max(candidate_diagonal, 1e-12)
    translation = reference_center - candidate_center * scale
    return scale, translation


def _residual(
    kind: str, magnitude: float, message: str, priority: int
) -> dict[str, Any]:
    return {
        "kind": kind,
        "normalizedMagnitude": float(max(0.0, magnitude)),
        "priority": int(priority),
        "message": message,
    }


def compare_meshes(
    reference: trimesh.Trimesh,
    candidate: trimesh.Trimesh,
    *,
    sample_count: int = 4096,
    seed: int = 42,
) -> dict[str, Any]:
    reference_points = _sample(reference, sample_count, seed)
    candidate_points_raw = _sample(candidate, sample_count, seed + 1)
    reference_bounds = np.asarray(reference.bounds, dtype=np.float64)
    candidate_bounds = np.asarray(candidate.bounds, dtype=np.float64)
    alignment_scale, translation = _alignment(reference_bounds, candidate_bounds)
    candidate_points = candidate_points_raw * alignment_scale + translation

    forward = _nearest(reference_points, candidate_points)
    reverse = _nearest(candidate_points, reference_points)
    all_distances = np.concatenate([forward, reverse])
    scale = max(
        float(np.linalg.norm(reference_bounds[1] - reference_bounds[0])), 1e-12
    )
    normalized_mean = float(all_distances.mean() / scale)
    normalized_p95 = float(np.percentile(all_distances, 95) / scale)

    reference_extents = reference_bounds[1] - reference_bounds[0]
    candidate_extents = candidate_bounds[1] - candidate_bounds[0]
    normalized_candidate_extents = candidate_extents * alignment_scale
    extent_errors = np.abs(normalized_candidate_extents - reference_extents) / np.maximum(
        np.abs(reference_extents), 1e-12
    )
    center_distance_raw = float(
        np.linalg.norm(candidate_bounds.mean(axis=0) - reference_bounds.mean(axis=0))
    )
    volume_reference = float(reference.volume) if reference.is_volume else None
    volume_candidate = float(candidate.volume) if candidate.is_volume else None
    area_error = _relative_error(
        float(reference.area), float(candidate.area) * alignment_scale**2
    )
    volume_error = _relative_error(
        volume_reference,
        None if volume_candidate is None else volume_candidate * alignment_scale**3,
    )
    reference_components = int(len(reference.split(only_watertight=False)))
    candidate_components = int(len(candidate.split(only_watertight=False)))

    residuals = [
        _residual(
            "surface",
            normalized_p95,
            f"95% of aligned sampled surface error is within {normalized_p95:.4f} reference diagonals.",
            100,
        ),
        *[
            _residual(
                f"extent-{axis}",
                float(error),
                f"Aligned {axis.upper()} extent differs by {float(error) * 100:.1f}%.",
                90,
            )
            for axis, error in zip(("x", "y", "z"), extent_errors)
        ],
    ]
    if area_error is not None:
        residuals.append(
            _residual(
                "surface-area",
                area_error,
                f"Aligned surface area differs by {area_error * 100:.1f}%.",
                60,
            )
        )
    if volume_error is not None:
        residuals.append(
            _residual(
                "volume",
                volume_error,
                f"Aligned enclosed volume differs by {volume_error * 100:.1f}%.",
                70,
            )
        )
    if reference_components != candidate_components:
        residuals.append(
            _residual(
                "component-count",
                abs(reference_components - candidate_components)
                / max(reference_components, 1),
                f"Reference has {reference_components} component(s); candidate has {candidate_components}.",
                95,
            )
        )
    if reference.is_watertight != candidate.is_watertight:
        residuals.append(
            _residual(
                "watertightness",
                1.0,
                "Reference and candidate disagree on watertightness.",
                80,
            )
        )
    actionable = sorted(
        (item for item in residuals if item["normalizedMagnitude"] >= 0.005),
        key=lambda item: (item["priority"], item["normalizedMagnitude"]),
        reverse=True,
    )[:12]
    similarity = float(
        np.clip(
            np.exp(-8.0 * normalized_p95)
            * np.exp(-2.0 * float(np.mean(extent_errors))),
            0.0,
            1.0,
        )
    )

    return {
        "version": COMPARISON_VERSION,
        "sampleCount": int(sample_count),
        "scale": scale,
        "alignment": {
            "mode": "center-and-uniform-scale",
            "candidateScale": float(alignment_scale),
            "candidateTranslation": translation.tolist(),
        },
        "similarityScore": similarity,
        "surfaceDistance": {
            "symmetricMean": float(all_distances.mean()),
            "symmetricRms": float(np.sqrt(np.mean(all_distances**2))),
            "p95": float(np.percentile(all_distances, 95)),
            "p99": float(np.percentile(all_distances, 99)),
            "maximum": float(all_distances.max()),
            "normalizedMean": normalized_mean,
            "normalizedP95": normalized_p95,
        },
        "bounds": {
            "referenceExtents": reference_extents.tolist(),
            "candidateExtents": candidate_extents.tolist(),
            "alignedCandidateExtents": normalized_candidate_extents.tolist(),
            "extentRelativeError": extent_errors.tolist(),
            "centerDistance": center_distance_raw,
        },
        "surfaceAreaRelativeError": area_error,
        "volumeRelativeError": volume_error,
        "topology": {
            "referenceWatertight": bool(reference.is_watertight),
            "candidateWatertight": bool(candidate.is_watertight),
            "referenceComponents": reference_components,
            "candidateComponents": candidate_components,
        },
        "actionableResiduals": actionable,
        "converged": bool(
            normalized_p95 <= 0.01
            and float(np.max(extent_errors)) <= 0.02
            and reference_components == candidate_components
        ),
    }


def compare_paths(
    reference_path: str, candidate_path: str, **options: Any
) -> dict[str, Any]:
    return compare_meshes(
        load_mesh(reference_path), load_mesh(candidate_path), **options
    )
