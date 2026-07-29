from __future__ import annotations

import contextlib
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .meshio import save_npz
from .profiles import ExecutionProfile


class GenerationUnavailable(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


class ShapeOnlyTrellisPipelineMixin:
    shape_model_names_to_load = [
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
    ]


def _load_pipeline(model: str, shape_only: bool):
    try:
        import torch
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
    except Exception as exc:
        raise GenerationUnavailable(
            "TRELLIS.2 inference dependencies are unavailable; run the setup script in WSL/Linux"
        ) from exc

    if shape_only:
        class ShapeOnlyPipeline(ShapeOnlyTrellisPipelineMixin, Trellis2ImageTo3DPipeline):
            model_names_to_load = ShapeOnlyTrellisPipelineMixin.shape_model_names_to_load

        pipeline = ShapeOnlyPipeline.from_pretrained(model)
    else:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model)
    if not torch.cuda.is_available():
        raise GenerationUnavailable("CUDA is required for TRELLIS.2 generation")
    pipeline.cuda()
    return pipeline


def _check(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise GenerationCancelled("generation cancelled")


def _shape_only_run(pipeline, image, profile: ExecutionProfile, seed: int, cancel: threading.Event):
    import torch

    _check(cancel)
    image = pipeline.preprocess_image(image)
    torch.manual_seed(seed)
    cond_512 = pipeline.get_cond([image], 512)
    cond_1024 = pipeline.get_cond([image], 1024) if profile.resolution != 512 else None
    _check(cancel)
    coords = pipeline.sample_sparse_structure(cond_512, 32, 1, {})
    _check(cancel)
    if profile.resolution == 512:
        shape_slat = pipeline.sample_shape_slat(
            cond_512, pipeline.models["shape_slat_flow_model_512"], coords, {}
        )
        resolution = 512
    else:
        shape_slat, resolution = pipeline.sample_shape_slat_cascade(
            cond_512,
            cond_1024,
            pipeline.models["shape_slat_flow_model_512"],
            pipeline.models["shape_slat_flow_model_1024"],
            512,
            profile.resolution,
            coords,
            {},
            profile.max_tokens,
        )
    _check(cancel)
    meshes, _ = pipeline.decode_shape_slat(shape_slat, resolution)
    torch.cuda.empty_cache()
    return meshes, shape_slat, resolution


def generate(
    *,
    image_path: str,
    output_directory: Path,
    profile: ExecutionProfile,
    model: str,
    seed: int,
    cancel: threading.Event,
) -> tuple[list[Path], dict[str, Any]]:
    try:
        from PIL import Image
    except Exception as exc:
        raise GenerationUnavailable("Pillow is required for image generation") from exc

    image_source = Path(image_path).expanduser().resolve()
    if not image_source.is_file():
        raise ValueError(f"input image does not exist: {image_source}")
    image = Image.open(image_source)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pipeline = _load_pipeline(model, profile.shape_only)
    _check(cancel)
    with contextlib.redirect_stdout(sys.stderr):
        if profile.shape_only:
            meshes, latent, resolution = _shape_only_run(pipeline, image, profile, seed, cancel)
        else:
            meshes, latent_payload = pipeline.run(
                image,
                seed=seed,
                pipeline_type="512" if profile.resolution == 512 else "1024_cascade",
                max_num_tokens=profile.max_tokens,
                return_latent=True,
            )
            latent = latent_payload[0]
            resolution = latent_payload[2]
    files: list[Path] = []
    for index, mesh in enumerate(meshes):
        vertices = mesh.vertices.detach().cpu().numpy() if hasattr(mesh.vertices, "detach") else np.asarray(mesh.vertices)
        faces = mesh.faces.detach().cpu().numpy() if hasattr(mesh.faces, "detach") else np.asarray(mesh.faces)
        files.append(save_npz(output_directory / f"shape-{index}.npz", vertices, faces))
    if hasattr(latent, "coords") and hasattr(latent, "feats"):
        latent_path = output_directory / "shape-latent.npz"
        np.savez_compressed(
            latent_path,
            coords=latent.coords.detach().cpu().numpy(),
            feats=latent.feats.detach().cpu().numpy(),
        )
        files.append(latent_path)
    return files, {
        "model": model,
        "profile": profile.name,
        "seed": seed,
        "resolution": int(resolution),
        "candidateCount": len(meshes),
        "shapeOnly": profile.shape_only,
    }
