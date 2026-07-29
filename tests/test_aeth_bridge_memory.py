from pathlib import Path

from aeth_bridge.generate import GenerationInsufficientMemory
from aeth_bridge.service import BridgeService


def test_insufficient_memory_carries_measurements(tmp_path: Path) -> None:
    error = GenerationInsufficientMemory(
        "out of memory",
        {
            "profile": "shape-512",
            "memoryBefore": {"totalMiB": 12288, "freeMiB": 11000},
            "memoryAtFailure": {"peakAllocatedMiB": 10950},
        },
    )
    assert error.details["profile"] == "shape-512"
    assert error.details["memoryBefore"]["totalMiB"] == 12288
    # Constructing the service is CPU-only and proves the error type remains
    # usable at the protocol boundary without importing TRELLIS or CUDA.
    service = BridgeService(tmp_path)
    service.executor.shutdown(wait=True)
