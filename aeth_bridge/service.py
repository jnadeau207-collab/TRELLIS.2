from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .analyze import analyze_path
from .artifacts import atomic_write_json, request_directory, safe_artifact_root, write_manifest
from .compare import compare_paths
from .generate import (
    GenerationCancelled,
    GenerationInsufficientMemory,
    GenerationUnavailable,
    generate,
)
from .profiles import probe, profile
from .protocol import ProtocolError, Request, event, parse_request, write_event


class BridgeService:
    def __init__(self, artifact_root: Path):
        self.artifact_root = artifact_root
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aeth-bridge")
        self.tasks: dict[str, tuple[Future[Any], threading.Event]] = {}
        self.write_lock = threading.Lock()

    def emit(self, value: dict[str, Any]) -> None:
        with self.write_lock:
            write_event(sys.stdout, value)

    def submit(self, request: Request) -> None:
        if request.request_id in self.tasks:
            self.emit(
                event(
                    request.request_id,
                    "error",
                    code="duplicate_request",
                    message="requestId is already active",
                )
            )
            return
        cancellation = threading.Event()
        future = self.executor.submit(self.execute, request, cancellation)
        self.tasks[request.request_id] = (future, cancellation)
        future.add_done_callback(lambda _: self.tasks.pop(request.request_id, None))

    def execute(self, request: Request, cancellation: threading.Event) -> None:
        self.emit(event(request.request_id, "started", command=request.command))
        try:
            if request.command == "probe":
                result = probe()
            elif request.command == "analyze":
                result = self.handle_analyze(request)
            elif request.command == "compare":
                result = self.handle_compare(request)
            elif request.command == "generate":
                result = self.handle_generate(request, cancellation)
            else:
                raise ProtocolError(f"command cannot run as task: {request.command}")
            self.emit(event(request.request_id, "completed", result=result))
        except GenerationCancelled as exc:
            self.emit(event(request.request_id, "cancelled", message=str(exc)))
        except GenerationInsufficientMemory as exc:
            self.emit(
                event(
                    request.request_id,
                    "error",
                    code="insufficient_vram",
                    message=str(exc),
                    details=exc.details,
                )
            )
        except GenerationUnavailable as exc:
            self.emit(
                event(
                    request.request_id,
                    "error",
                    code="generation_unavailable",
                    message=str(exc),
                )
            )
        except Exception as exc:
            self.emit(
                event(
                    request.request_id,
                    "error",
                    code="execution_failed",
                    message=str(exc),
                    details={"type": type(exc).__name__},
                )
            )
            traceback.print_exc(file=sys.stderr)

    def handle_analyze(self, request: Request) -> dict[str, Any]:
        source = request.payload.get("path")
        if not isinstance(source, str):
            raise ValueError("analyze.path must be a string")
        directory = request_directory(self.artifact_root, request.request_id)
        report = analyze_path(source)
        report_path = directory / "analysis.json"
        atomic_write_json(report_path, report)
        manifest_path = write_manifest(
            directory,
            request_id=request.request_id,
            operation="analyze",
            files=[report_path],
            metadata={"source": source},
        )
        return {
            "reportPath": str(report_path),
            "manifestPath": str(manifest_path),
            "report": report,
        }

    def handle_compare(self, request: Request) -> dict[str, Any]:
        reference = request.payload.get("referencePath")
        candidate = request.payload.get("candidatePath")
        if not isinstance(reference, str) or not isinstance(candidate, str):
            raise ValueError(
                "compare requires referencePath and candidatePath strings"
            )
        sample_count = int(request.payload.get("sampleCount", 4096))
        seed = int(request.payload.get("seed", 42))
        directory = request_directory(self.artifact_root, request.request_id)
        comparison = compare_paths(
            reference, candidate, sample_count=sample_count, seed=seed
        )
        comparison_path = directory / "comparison.json"
        atomic_write_json(comparison_path, comparison)
        manifest_path = write_manifest(
            directory,
            request_id=request.request_id,
            operation="compare",
            files=[comparison_path],
            metadata={"reference": reference, "candidate": candidate},
        )
        return {
            "comparisonPath": str(comparison_path),
            "manifestPath": str(manifest_path),
            "comparison": comparison,
        }

    def handle_generate(
        self, request: Request, cancellation: threading.Event
    ) -> dict[str, Any]:
        image_path = request.payload.get("imagePath")
        profile_name = request.payload.get("profile", "shape-512")
        model = request.payload.get("model", "microsoft/TRELLIS.2-4B")
        seed = int(request.payload.get("seed", 42))
        if not isinstance(image_path, str):
            raise ValueError("generate.imagePath must be a string")
        if not isinstance(profile_name, str) or not isinstance(model, str):
            raise ValueError("generate.profile and generate.model must be strings")
        directory = request_directory(self.artifact_root, request.request_id)
        files, metadata = generate(
            image_path=image_path,
            output_directory=directory,
            profile=profile(profile_name),
            model=model,
            seed=seed,
            cancel=cancellation,
        )
        manifest_path = write_manifest(
            directory,
            request_id=request.request_id,
            operation="generate",
            files=files,
            metadata=metadata,
        )
        return {
            "manifestPath": str(manifest_path),
            "artifacts": [str(path) for path in files],
            "metadata": metadata,
        }

    def cancel(self, request: Request) -> None:
        target = request.payload.get("targetRequestId")
        if not isinstance(target, str):
            self.emit(
                event(
                    request.request_id,
                    "error",
                    code="invalid_cancel",
                    message="targetRequestId must be a string",
                )
            )
            return
        active = self.tasks.get(target)
        if active is None:
            self.emit(
                event(
                    request.request_id,
                    "completed",
                    result={"cancelled": False, "reason": "not_active"},
                )
            )
            return
        active[1].set()
        self.emit(
            event(
                request.request_id,
                "completed",
                result={"cancelled": True, "targetRequestId": target},
            )
        )

    def run(self) -> int:
        self.emit(
            event(
                "bridge",
                "ready",
                result={"artifactRoot": str(self.artifact_root)},
            )
        )
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = parse_request(line)
            except ProtocolError as exc:
                self.emit(
                    event(
                        "unknown",
                        "error",
                        code="protocol_error",
                        message=str(exc),
                    )
                )
                continue
            if request.command == "cancel":
                self.cancel(request)
            elif request.command == "shutdown":
                for _, cancellation in self.tasks.values():
                    cancellation.set()
                self.emit(
                    event(
                        request.request_id,
                        "completed",
                        result={"shutdown": True},
                    )
                )
                break
            else:
                self.submit(request)
        self.executor.shutdown(wait=True, cancel_futures=False)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 deterministic stdio bridge"
    )
    parser.add_argument(
        "--artifact-root",
        default=os.environ.get(
            "AETH_BRIDGE_ARTIFACT_ROOT",
            str(Path.home() / ".cache" / "aeth-bridge"),
        ),
    )
    parser.add_argument(
        "--probe", action="store_true", help="print a one-shot JSON capability probe"
    )
    arguments = parser.parse_args(argv)
    if arguments.probe:
        json.dump(probe(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    return BridgeService(safe_artifact_root(arguments.artifact_root)).run()
