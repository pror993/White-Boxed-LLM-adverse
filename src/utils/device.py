"""Device management and Apple Silicon MPS memory tracking utilities.

Provides robust Metal (MPS) device discovery, memory introspection,
driver/unified memory telemetry, safe allocation contexts, and CPU fallbacks
tailored for 16GB Apple Silicon architectures.
"""

from __future__ import annotations

import gc
import logging
import os
import platform
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, Optional, Tuple, Union

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceConfig:
    """Hardware and execution backend configuration."""

    preferred_device: str = "mps"
    fallback_device: str = "cpu"
    default_dtype: str = "float16"
    enable_memory_efficient_ops: bool = True
    mps_memory_fraction: Optional[float] = 0.75  # ~12GB of 16GB unified memory


@dataclass
class MPSMemorySnapshot:
    """Snapshot of Metal Performance Shaders and system unified memory state."""

    timestamp: float
    device: str
    allocated_bytes: int
    driver_allocated_bytes: int
    process_rss_bytes: int
    system_available_bytes: int
    system_total_bytes: int

    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / (1024 * 1024)

    @property
    def driver_allocated_mb(self) -> float:
        return self.driver_allocated_bytes / (1024 * 1024)

    @property
    def process_rss_mb(self) -> float:
        return self.process_rss_bytes / (1024 * 1024)

    @property
    def system_available_mb(self) -> float:
        return self.system_available_bytes / (1024 * 1024)

    @property
    def system_total_mb(self) -> float:
        return self.system_total_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "allocated_mb": round(self.allocated_mb, 2),
                "driver_allocated_mb": round(self.driver_allocated_mb, 2),
                "process_rss_mb": round(self.process_rss_mb, 2),
                "system_available_mb": round(self.system_available_mb, 2),
                "system_total_mb": round(self.system_total_mb, 2),
            }
        )
        return data

    def __repr__(self) -> str:
        return (
            f"MPSMemorySnapshot(device={self.device}, "
            f"allocated={self.allocated_mb:.2f}MB, "
            f"driver={self.driver_allocated_mb:.2f}MB, "
            f"process_rss={self.process_rss_mb:.2f}MB, "
            f"sys_avail={self.system_available_mb:.2f}MB)"
        )


def is_apple_silicon() -> bool:
    """Check if the current host machine is running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def is_mps_available() -> bool:
    """Verify if Apple Silicon Metal Performance Shaders (MPS) is available and functional."""
    if not hasattr(torch.backends, "mps"):
        return False
    if not torch.backends.mps.is_available():
        return False
    if not torch.backends.mps.is_built():
        return False

    # Execute a lightweight allocation test to confirm functional Metal runtime
    try:
        test_tensor = torch.zeros(1, device="mps")
        del test_tensor
        return True
    except Exception as e:
        logger.warning("MPS reported available but probe allocation failed: %s", e)
        return False


def get_best_device(preferred: Optional[str] = "mps", fallback: str = "cpu") -> torch.device:
    """Resolve optimal compute device with graceful fallback.

    Args:
        preferred: Desired device identifier ('mps', 'cuda', or 'cpu').
        fallback: Safe fallback device identifier when preferred is unavailable.

    Returns:
        torch.device instance.
    """
    if preferred == "mps" and is_mps_available():
        return torch.device("mps")
    elif preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred == "cpu":
        return torch.device("cpu")

    logger.info("Preferred device '%s' unavailable. Falling back to '%s'.", preferred, fallback)
    return torch.device(fallback)


def get_device_config(preferred: str = "mps") -> DeviceConfig:
    """Construct standard DeviceConfig tuned for local environment."""
    return DeviceConfig(
        preferred_device=preferred if is_mps_available() else "cpu",
        fallback_device="cpu",
        default_dtype="float16" if is_mps_available() else "float32",
    )


def synchronize_device(device: Optional[Union[torch.device, str]] = None) -> None:
    """Synchronize device execution stream to guarantee completion of prior kernels."""
    dev = torch.device(device) if device is not None else get_best_device()
    if dev.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def clear_device_cache(device: Optional[Union[torch.device, str]] = None) -> None:
    """Explicitly release Metal / GPU cached buffers and execute Python garbage collection."""
    gc.collect()
    dev = torch.device(device) if device is not None else get_best_device()
    if dev.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def get_mps_memory_snapshot(device: Optional[Union[torch.device, str]] = None) -> MPSMemorySnapshot:
    """Capture precise memory snapshot across MPS driver and system unified memory."""
    dev = torch.device(device) if device is not None else get_best_device()
    now = time.time()

    allocated_bytes = 0
    driver_allocated_bytes = 0

    if dev.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "current_allocated_memory"):
            allocated_bytes = torch.mps.current_allocated_memory()
        if hasattr(torch.mps, "driver_allocated_memory"):
            driver_allocated_bytes = torch.mps.driver_allocated_memory()
    elif dev.type == "cuda":
        allocated_bytes = torch.cuda.memory_allocated()
        driver_allocated_bytes = torch.cuda.memory_reserved()

    # System unified memory metrics via psutil
    proc = psutil.Process(os.getpid())
    process_rss = proc.memory_info().rss
    vm = psutil.virtual_memory()

    return MPSMemorySnapshot(
        timestamp=now,
        device=str(dev),
        allocated_bytes=allocated_bytes,
        driver_allocated_bytes=driver_allocated_bytes,
        process_rss_bytes=process_rss,
        system_available_bytes=vm.available,
        system_total_bytes=vm.total,
    )


class MPSMemoryTracker:
    """Context manager and tracker for profiling peak memory usage of tensor operations."""

    def __init__(self, name: str = "operation", device: Optional[Union[torch.device, str]] = None):
        self.name = name
        self.device = torch.device(device) if device is not None else get_best_device()
        self.start_snapshot: Optional[MPSMemorySnapshot] = None
        self.end_snapshot: Optional[MPSMemorySnapshot] = None
        self.peak_allocated_mb: float = 0.0

    def __enter__(self) -> MPSMemoryTracker:
        synchronize_device(self.device)
        clear_device_cache(self.device)
        self.start_snapshot = get_mps_memory_snapshot(self.device)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        synchronize_device(self.device)
        self.end_snapshot = get_mps_memory_snapshot(self.device)

    @property
    def delta_allocated_mb(self) -> float:
        if self.start_snapshot is None or self.end_snapshot is None:
            return 0.0
        return self.end_snapshot.allocated_mb - self.start_snapshot.allocated_mb

    @property
    def delta_driver_mb(self) -> float:
        if self.start_snapshot is None or self.end_snapshot is None:
            return 0.0
        return self.end_snapshot.driver_allocated_mb - self.start_snapshot.driver_allocated_mb

    @property
    def delta_rss_mb(self) -> float:
        if self.start_snapshot is None or self.end_snapshot is None:
            return 0.0
        return self.end_snapshot.process_rss_mb - self.start_snapshot.process_rss_mb

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device": str(self.device),
            "start_allocated_mb": self.start_snapshot.allocated_mb if self.start_snapshot else 0.0,
            "end_allocated_mb": self.end_snapshot.allocated_mb if self.end_snapshot else 0.0,
            "delta_allocated_mb": round(self.delta_allocated_mb, 2),
            "delta_driver_mb": round(self.delta_driver_mb, 2),
            "delta_rss_mb": round(self.delta_rss_mb, 2),
        }


@contextmanager
def safe_mps_context(
    clear_cache_on_entry: bool = True,
    clear_cache_on_exit: bool = True,
    device: Optional[Union[torch.device, str]] = None,
) -> Generator[None, None, None]:
    """Context manager ensuring memory hygiene and diagnostic logging during MPS operations."""
    dev = torch.device(device) if device is not None else get_best_device()
    if clear_cache_on_entry:
        clear_device_cache(dev)
    try:
        yield
    except RuntimeError as e:
        err_msg = str(e).lower()
        if "out of memory" in err_msg or "mps" in err_msg or "metal" in err_msg:
            logger.error("MPS Error during execution: %s", e)
            clear_device_cache(dev)
            snapshot = get_mps_memory_snapshot(dev)
            logger.error("Memory State after failure: %s", snapshot)
        raise
    finally:
        if clear_cache_on_exit:
            clear_device_cache(dev)


def estimate_safe_micro_batch_size(
    model_param_count: int,
    seq_len: int,
    bytes_per_param: int = 2,  # float16/bfloat16
    safety_margin_gb: float = 4.0,
    total_memory_gb: float = 16.0,
    max_batch: int = 128,
) -> int:
    """Heuristic estimator for candidate evaluation micro-batch size within unified memory.

    Args:
        model_param_count: Total parameters in the model (e.g., 1.1e9 for Llama-3.2-1B).
        seq_len: Input sequence length including prompt and target tokens.
        bytes_per_param: Precision footprint (2 for fp16/bf16, 4 for fp32).
        safety_margin_gb: Reserved overhead memory for OS and framework buffers.
        total_memory_gb: Host physical unified memory.
        max_batch: Upper ceiling for micro-batch size.

    Returns:
        Recommended integer micro-batch size.
    """
    model_weight_gb = (model_param_count * bytes_per_param) / (1024**3)
    available_gb = max(1.0, total_memory_gb - model_weight_gb - safety_margin_gb)

    # Approximate activation memory per candidate sequence
    # Per-candidate memory includes KV cache, intermediate layer activations, and logits
    # Conservative rule of thumb: ~4 * seq_len * hidden_size * num_layers * bytes_per_elem
    # For ~1B-3B models, activation per sequence is typically ~10-40 MB
    estimated_mb_per_candidate = max(5.0, (seq_len * 2048 * 24 * bytes_per_param * 2) / (1024**2))
    candidate_capacity = int((available_gb * 1024) / estimated_mb_per_candidate)

    safe_batch = max(1, min(max_batch, candidate_capacity))
    # Round down to power of 2 or multiple of 8 if >= 8
    if safe_batch >= 8:
        safe_batch = (safe_batch // 8) * 8
    return safe_batch
