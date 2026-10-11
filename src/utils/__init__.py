"""Hardware, device management, and memory tracking utilities for Apple Silicon MPS."""

from src.utils.device import (
    DeviceConfig,
    MPSMemorySnapshot,
    MPSMemoryTracker,
    get_best_device,
    get_device_config,
    get_mps_memory_snapshot,
    is_mps_available,
    synchronize_device,
    clear_device_cache,
    safe_mps_context,
)

__all__ = [
    "DeviceConfig",
    "MPSMemorySnapshot",
    "MPSMemoryTracker",
    "get_best_device",
    "get_device_config",
    "get_mps_memory_snapshot",
    "is_mps_available",
    "synchronize_device",
    "clear_device_cache",
    "safe_mps_context",
]
