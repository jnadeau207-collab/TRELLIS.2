import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from aeth_bridge import profiles
from aeth_bridge.analyze import analyze_path
from aeth_bridge.artifacts import build_manifest, request_directory, safe_artifact_root
from aeth_bridge.compare import compare_paths
from aeth_bridge.meshio import resolve_input_path
from aeth_bridge.protocol import PROTOCOL_VERSION, ProtocolError, parse_request


def _save(path: Path, mesh: trimesh.Trimesh) -> None:
    np.savez_compressed(
        path, vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)
    )


def test_parse_request() -> None:
    request = parse_request(
        json.dumps(
            {
                "version": PROTOCOL_VERSION,
                "requestId": "r1",
                "command": "probe",
                "payload": {},
            }
        )
    )
    assert request.request_id == "r1"
    assert request.command == "probe"


def test_rejects_unknown_version() -> None:
    with pytest.raises(ProtocolError):
        parse_request('{"version":99,"requestId":"r1","command":"probe"}')


def test_manifest_hashes_files(tmp_path: Path) -> None:
    root = safe_artifact_root(tmp_path)
    directory = request_directory(root, "request")
    item = directory / "item.bin"
    item.write_bytes(b"abc")
    manifest = build_manifest(
        directory, request_id="request", operation="test", files=[item]
    )
    assert (
        manifest["files"][0]["sha256"]
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_analysis_and_comparison(tmp_path: Path) -> None:
    reference = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    candidate = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    reference_path = tmp_path / "reference.npz"
    candidate_path = tmp_path / "candidate.npz"
    _save(reference_path, reference)
    _save(candidate_path, candidate)
    report = analyze_path(str(reference_path))
    assert report["mesh"]["watertight"] is True
    assert report["bounds"]["extents"] == [2.0, 3.0, 4.0]
    comparison = compare_paths(
        str(reference_path), str(candidate_path), sample_count=512, seed=7
    )
    assert comparison["surfaceDistance"]["normalizedMean"] < 0.08
    assert comparison["volumeRelativeError"] == 0.0


def test_windows_paths_are_translated_inside_wsl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    assert resolve_input_path(r"C:\designs\reference.png") == Path(
        "/mnt/c/designs/reference.png"
    )


def test_probe_reports_candidate_profiles_without_guessing_a_vram_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        profiles,
        "_nvidia",
        lambda: [
            {
                "name": "Test GPU",
                "memoryMiB": 12_288,
                "memoryFreeMiB": 10_240,
                "driverVersion": "test",
            }
        ],
    )
    monkeypatch.setattr(profiles, "_module", lambda _name: True)
    monkeypatch.setattr(
        profiles,
        "_cuda_state",
        lambda: {
            "available": True,
            "deviceCount": 1,
            "torchVersion": "test",
            "cudaVersion": "test",
        },
    )
    result = profiles.probe()
    assert result["recommendedProfile"] == "shape-512"
    assert result["qualificationPolicy"] == "attempt-and-measure"
    assert "shape-512" in result["availableProfiles"]
    assert "minimum_vram_gib" not in result["profiles"]["shape-512"]


def test_stdio_probe_and_shutdown(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "aeth_bridge", "--artifact-root", str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    ready = json.loads(process.stdout.readline())
    assert ready["kind"] == "ready"
    process.stdin.write(
        '{"version":1,"requestId":"probe-1","command":"probe","payload":{}}\n'
    )
    process.stdin.flush()
    started = json.loads(process.stdout.readline())
    completed = json.loads(process.stdout.readline())
    assert started["kind"] == "started"
    assert completed["kind"] == "completed"
    assert completed["result"]["qualificationPolicy"] == "attempt-and-measure"
    process.stdin.write(
        '{"version":1,"requestId":"stop","command":"shutdown","payload":{}}\n'
    )
    process.stdin.flush()
    assert json.loads(process.stdout.readline())["kind"] == "completed"
    assert process.wait(timeout=15) == 0
