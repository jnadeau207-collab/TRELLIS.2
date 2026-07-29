from __future__ import annotations

import contextlib
import gc
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


class GenerationInsufficientMemory(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


class GenerationCancelled(RuntimeError):
    pass


class ShapeOnlyTrellisPipelineMixin:
    # Texture flow/decoder weights are intentionally absent.  This is the
    # largest memory reduction available without changing the released model.
    shape_model_names_to_load = [
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
    ]


def _memory_snapshot(torch: Any) -> dict[str, int]:
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "freeMiB": int(free_bytes // (1024 * 1024)),
            "totalMiB": int(total_bytes // (1024 * 1024)),
            "allocatedMiB": int(torch.cuda.memory_allocated() // (1024 * 1024)),
            "reservedMiB": int(torch.cuda.memory_reserved() // (1024 * 1024)),
            "peakAllocatedMiB": int(torch.cuda.max_memory_allocated() // (1024 * 1024)),
            "peakReservedMiB": int(torch.cuda.max_memory_reserved() // (1024 * 1024)),
        }
    except Exception:
        return {}


def _is_cuda_oom(error: BaseException, torch: Any | None) -> bool:
    if torch is not None and isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()


def _load_pipeline(model: str, profile: ExecutionProfile):
    try:
        import torch
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
    except Exception as exc:
        raise GenerationUnavailable(
            "TRELLIS.2 inference dependencies are unavailable; run the setup script in WSL/Linux"
        ) from exc

    if not torch.cuda.is_available():
        raise GenerationUnavailable("CUDA is required for TRELLIS.2 generation")

    if profile.shape_only:

        class ShapeOnlyPipeline(
            ShapeOnlyTrellisPipelineMixin, Trellis2ImageTo3DPipeline
        ):
            model_names_to_load = ShapeOnlyTrellisPipelineMixin.shape_model_names_to_load

        pipeline = ShapeOnlyPipeline.from_pretrained(model)
    else:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model)

    # The upstream pipeline already implements one-model-at-a-time CUDA
    # residency.  Force it on for every bridge profile and set only the target
    # device; do not move the entire model dictionary to the GPU.
    pipeline.low_vram = profile.low_vram
    pipeline.to(torch.device("cuda"))
    return pipeline


def _check(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise GenerationCancelled("generation cancelled")


def _shape_only_run(
    pipeline: Any,
    image: Any,
    profile: ExecutionProfile,
    seed: int,
    cancel: threading.Event,
):
    import torch

    _check(cancel)
    image = pipeline.preprocess_image(image)
    torch.manual_seed(seed)
    cond_512 = pipeline.get_cond([image], 512)
    cond_1024 = (
        pipeline.get_cond([image], 1024) if profile.resolution != 512 else None
    )
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
        import torch
    except Exception as exc:
        raise GenerationUnavailable(
            "PyTorch and Pillow are required for image generation"
        ) from exc

    image_source = Path(image_path).expanduser().resolve()
    if not image_source.is_file():
        raise ValueError(f"input image does not exist: {image_source}")

    # Fragmentation controls matter on 12 GiB-class cards.  The setting is a
    # default so an operator can still override it before process launch.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )
    if not torch.cuda.is_available():
        raise GenerationUnavailable("CUDA is required for TRELLIS.2 generation")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    memory_before = _memory_snapshot(torch)
    pipeline: Any | None = None
    try:
        image = Image.open(image_source)
        pipeline = _load_pipeline(model, profile)
        _check(cancel)
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            if profile.shape_only:
                meshes, latent, resolution = _shape_only_run(
                    pipeline, image, profile, seed, cancel
                )
            else:
                meshes, latent_payload = pipeline.run(
                    image,
                    seed=seed,
                    pipeline_type=(
                        "512" if profile.resolution == 512 else "1024_cascade"
                    ),
                    max_num_tokens=profile.max_tokens,
                    return_latent=True,
                )
                latent = latent_payload[0]
                resolution = latent_payload[2]
        _check(cancel)
        files: list[Path] = []
        for index, mesh in enumerate(meshes):
            vertices = (
                mesh.vertices.detach().cpu().numpy()
                if hasattr(mesh.vertices, "detach")
                else np.asarray(mesh.vertices)
            )
            faces = (
                mesh.faces.detach().cpu().numpy()
                if hasattr(mesh.faces, "detach")
                else np.asarray(mesh.faces)
            )
            files.append(
                save_npz(
                    output_directory / f"shape-{index}.npz", vertices, faces
                )
            )
        if hasattr(latent, "coords") and hasattr(latent, "feats"):
            latent_path = output_directory / "shape-latent.npz"
            np.savez_compressed(
                latent_path,
                coords=latent.coords.detach().cpu().numpy(),
                feats=latent.feats.detach().cpu().numpy(),
            )
            files.append(latent_path)
        memory_after = _memory_snapshot(torch)
        return files, {
            "model": model,
            "profile": profile.name,
            "seed": seed,
            "resolution": int(resolution),
            "candidateCount": len(meshes),
            "shapeOnly": profile.shape_only,
            "lowVram": profile.low_vram,
            "memoryBefore": memory_before,
            "memoryAfter": memory_after,
        }
    except GenerationCancelled:
        raise
    except Exception as exc:
        if _is_cuda_oom(exc, torch):
            details = {
                "profile": profile.name,
                "memoryBefore": memory_before,
                "memoryAtFailure": _memory_snapshot(torch),
            }
            raise GenerationInsufficientMemory(
                "TRELLIS.2 exhausted CUDA memory in the measured low-VRAM run. "
                "Close other GPU workloads or use a remote GPU; no static VRAM "
                "threshold was used to reach this result.",
                details,
            ) from exc
        raise
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
