from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    resolution: int
    shape_only: bool
    max_tokens: int
    low_vram: bool
    description: str


# Profiles are execution strategies, not hardware promises.  In particular,
# shape-512 deliberately uses TRELLIS.2's low-VRAM model shuttling and loads no
# texture models.  A probe can establish CUDA/dependency prerequisites, but only
# an actual generation run can qualify a specific GPU/driver/input combination.
PROFILES = {
    "shape-512": ExecutionProfile(
        "shape-512",
        512,
        True,
        24_576,
        True,
        "Shape-only 512 pipeline with one-model-at-a-time CUDA residency; first choice for 12 GiB-class GPUs.",
    ),
    "shape-1024": ExecutionProfile(
        "shape-1024",
        1024,
        True,
        32_768,
        True,
        "Shape-only cascade with bounded high-resolution tokens and low-VRAM model shuttling.",
    ),
    "full-512": ExecutionProfile(
        "full-512",
        512,
        False,
        24_576,
        True,
        "Geometry and material generation at 512 with low-VRAM model shuttling.",
    ),
    "full-1024": ExecutionProfile(
        "full-1024",
        1024,
        False,
        49_152,
        True,
        "Geometry and material cascade at 1024; highest memory demand.",
    ),
}


def _run(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        return subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _nvidia() -> list[dict[str, Any]]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None or result.returncode != 0:
        return []
    output: list[dict[str, Any]] = []
    for row in result.stdout.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 4:
            continue
        try:
            total_mib = int(parts[1])
            free_mib = int(parts[2])
        except ValueError:
            total_mib = 0
            free_mib = 0
        output.append(
            {
                "name": parts[0],
                "memoryMiB": total_mib,
                "memoryFreeMiB": free_mib,
                "driverVersion": parts[3],
            }
        )
    return output


def _module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _cuda_state() -> dict[str, Any]:
    try:
        import torch

        return {
            "available": bool(torch.cuda.is_available()),
            "deviceCount": int(torch.cuda.device_count()),
            "torchVersion": str(torch.__version__),
            "cudaVersion": torch.version.cuda,
        }
    except Exception:
        return {
            "available": False,
            "deviceCount": 0,
            "torchVersion": None,
            "cudaVersion": None,
        }


def probe() -> dict[str, Any]:
    gpus = _nvidia()
    modules = {
        name: _module(name)
        for name in ("torch", "numpy", "trimesh", "PIL", "trellis2")
    }
    cuda = _cuda_state()
    wsl = bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in platform.release().lower()
    generation_prerequisites = (
        modules["torch"]
        and modules["PIL"]
        and modules["trellis2"]
        and cuda["available"]
        and bool(gpus)
    )
    candidate_profiles = list(PROFILES) if generation_prerequisites else []
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "wsl": wsl,
        },
        "gpus": gpus,
        "cuda": cuda,
        "modules": modules,
        "profiles": {name: asdict(value) for name, value in PROFILES.items()},
        # Kept for protocol compatibility.  "Available" means all software/CUDA
        # prerequisites exist, not that a guessed VRAM threshold blessed the run.
        "availableProfiles": candidate_profiles,
        "recommendedProfile": "shape-512" if generation_prerequisites else None,
        "qualificationPolicy": "attempt-and-measure",
        "capabilities": {
            "analysis": modules["numpy"] and modules["trimesh"],
            "generation": generation_prerequisites,
            "windowsHostSupported": True,
            "recommendedWindowsBackend": "wsl" if platform.system() == "Windows" else "local",
        },
    }


def profile(name: str) -> ExecutionProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown execution profile: {name}") from exc
