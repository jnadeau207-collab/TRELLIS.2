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


def compare_meshes(
    reference: trimesh.Trimesh,
    candidate: trimesh.Trimesh,
    *,
    sample_count: int = 4096,
    seed: int = 42,
) -> dict[str, Any]:
    reference_points = _sample(reference, sample_count, seed)
    candidate_points = _sample(candidate, sample_count, seed + 1)
    forward = _nearest(reference_points, candidate_points)
    reverse = _nearest(candidate_points, reference_points)
    all_distances = np.concatenate([forward, reverse])
    reference_bounds = np.asarray(reference.bounds, dtype=np.float64)
    candidate_bounds = np.asarray(candidate.bounds, dtype=np.float64)
    scale = max(float(np.linalg.norm(reference_bounds[1] - reference_bounds[0])), 1e-12)
    volume_reference = float(reference.volume) if reference.is_volume else None
    volume_candidate = float(candidate.volume) if candidate.is_volume else None
    return {
        "version": COMPARISON_VERSION,
        "sampleCount": int(sample_count),
        "scale": scale,
        "surfaceDistance": {
            "symmetricMean": float(all_distances.mean()),
            "symmetricRms": float(np.sqrt(np.mean(all_distances**2))),
            "p95": float(np.percentile(all_distances, 95)),
            "p99": float(np.percentile(all_distances, 99)),
            "maximum": float(all_distances.max()),
            "normalizedMean": float(all_distances.mean() / scale),
            "normalizedP95": float(np.percentile(all_distances, 95) / scale),
        },
        "bounds": {
            "referenceExtents": (reference_bounds[1] - reference_bounds[0]).tolist(),
            "candidateExtents": (candidate_bounds[1] - candidate_bounds[0]).tolist(),
            "extentRelativeError": (
                np.abs(
                    (candidate_bounds[1] - candidate_bounds[0])
                    - (reference_bounds[1] - reference_bounds[0])
                )
                / np.maximum(np.abs(reference_bounds[1] - reference_bounds[0]), 1e-12)
            ).tolist(),
            "centerDistance": float(
                np.linalg.norm(candidate_bounds.mean(axis=0) - reference_bounds.mean(axis=0))
            ),
        },
        "surfaceAreaRelativeError": _relative_error(float(reference.area), float(candidate.area)),
        "volumeRelativeError": _relative_error(volume_reference, volume_candidate),
        "topology": {
            "referenceWatertight": bool(reference.is_watertight),
            "candidateWatertight": bool(candidate.is_watertight),
            "referenceComponents": int(len(reference.split(only_watertight=False))),
            "candidateComponents": int(len(candidate.split(only_watertight=False))),
        },
    }


def compare_paths(reference_path: str, candidate_path: str, **options: Any) -> dict[str, Any]:
    return compare_meshes(load_mesh(reference_path), load_mesh(candidate_path), **options)
