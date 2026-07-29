from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

MANIFEST_VERSION = 1


class ArtifactError(ValueError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_artifact_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ArtifactError(f"artifact root is not a directory: {root}")
    return root


def request_directory(root: Path, request_id: str) -> Path:
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:20]
    directory = (root / digest).resolve()
    if root != directory and root not in directory.parents:
        raise ArtifactError("artifact path escaped root")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_manifest(
    directory: Path,
    *,
    request_id: str,
    operation: str,
    files: Iterable[Path],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entries = []
    for path in sorted(files, key=lambda item: item.name):
        resolved = path.resolve()
        if directory != resolved and directory not in resolved.parents:
            raise ArtifactError(f"file escaped artifact directory: {path}")
        entries.append(
            {
                "name": resolved.relative_to(directory).as_posix(),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return {
        "version": MANIFEST_VERSION,
        "requestId": request_id,
        "operation": operation,
        "files": entries,
        "metadata": metadata or {},
    }


def write_manifest(
    directory: Path,
    *,
    request_id: str,
    operation: str,
    files: Iterable[Path],
    metadata: dict[str, Any] | None = None,
) -> Path:
    manifest_path = directory / "manifest.json"
    manifest = build_manifest(
        directory,
        request_id=request_id,
        operation=operation,
        files=files,
        metadata=metadata,
    )
    atomic_write_json(manifest_path, manifest)
    return manifest_path
