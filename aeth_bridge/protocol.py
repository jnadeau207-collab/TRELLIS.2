from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TextIO

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 1_048_576
SUPPORTED_COMMANDS = frozenset(
    {"probe", "generate", "analyze", "compare", "cancel", "shutdown"}
)


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Request:
    request_id: str
    command: str
    payload: dict[str, Any]


def parse_request(line: str) -> Request:
    if len(line.encode("utf-8")) > MAX_LINE_BYTES:
        raise ProtocolError("request exceeds maximum line size")
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("request must be a JSON object")
    if raw.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {raw.get('version')!r}")
    request_id = raw.get("requestId")
    command = raw.get("command")
    payload = raw.get("payload", {})
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise ProtocolError("requestId must be a non-empty string of at most 128 characters")
    if command not in SUPPORTED_COMMANDS:
        raise ProtocolError(f"unsupported command: {command!r}")
    if not isinstance(payload, dict):
        raise ProtocolError("payload must be an object")
    return Request(request_id=request_id, command=command, payload=payload)


def event(request_id: str, kind: str, **data: Any) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "requestId": request_id,
        "kind": kind,
        **data,
    }


def write_event(stream: TextIO, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")
    stream.flush()
