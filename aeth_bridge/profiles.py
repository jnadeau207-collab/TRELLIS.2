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
    minimum_vram_gib: int


PROFILES = {
    "shape-512": ExecutionProfile("shape-512", 512, True, 24_576, 24),
    "shape-1024": ExecutionProfile("shape-1024", 1024, True, 49_152, 24),
    "full-512": ExecutionProfile("full-512", 512, False, 24_576, 24),
    "full-1024": ExecutionProfile("full-1024", 1024, False, 49_152, 32),
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
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None or result.returncode != 0:
        return []
    output = []
    for row in result.stdout.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 3:
            continue
        try:
            memory_mib = int(parts[1])
        except ValueError:
            memory_mib = 0
        output.append(
            {"name": parts[0], "memoryMiB": memory_mib, "driverVersion": parts[2]}
        )
    return output


def _module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def probe() -> dict[str, Any]:
    gpus = _nvidia()
    modules = {
        name: _module(name)
        for name in ("torch", "numpy", "trimesh", "PIL", "trellis2")
    }
    wsl = bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in platform.release().lower()
    maximum_vram = max((gpu["memoryMiB"] for gpu in gpus), default=0) / 1024
    available_profiles = [
        name for name, value in PROFILES.items() if maximum_vram >= value.minimum_vram_gib
    ]
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "wsl": wsl,
        },
        "gpus": gpus,
        "modules": modules,
        "profiles": {name: asdict(value) for name, value in PROFILES.items()},
        "availableProfiles": available_profiles,
        "capabilities": {
            "analysis": modules["numpy"] and modules["trimesh"],
            "generation": modules["torch"] and modules["PIL"] and modules["trellis2"] and bool(gpus),
            "windowsHostSupported": True,
            "recommendedWindowsBackend": "wsl" if platform.system() == "Windows" else "local",
        },
    }


def profile(name: str) -> ExecutionProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown execution profile: {name}") from exc
