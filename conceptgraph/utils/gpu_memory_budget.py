"""NVML-backed guard for the inference process's shared CUDA memory budget."""

from __future__ import annotations

import os

import torch


GIB = 1024**3
MAX_USED_BYTES = 28 * GIB
MIN_FREE_BYTES = 2 * GIB
MIN_STARTUP_FREE_BYTES = 30 * GIB


class NvmlMemoryBudget:
    """Enforce a 28 GiB device-use ceiling with 2 GiB of free headroom.

    NVML is required. Missing NVML support or unreadable device telemetry is a
    hard error so callers cannot silently run without the configured guard.
    """

    def __init__(self, device: torch.device):
        if device.type != "cuda":
            raise ValueError("NVML memory budget requires a CUDA device")
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError(
                "NVML telemetry is required for GPU inference; install the "
                "project's nvidia-ml-py dependency"
            ) from exc

        self._nvml = pynvml
        try:
            pynvml.nvmlInit()
            device_index = (
                device.index if device.index is not None else torch.cuda.current_device()
            )
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            total = int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).total)
        except Exception as exc:
            raise RuntimeError(f"Could not initialize NVML memory guard: {exc}") from exc
        if total <= MAX_USED_BYTES + MIN_FREE_BYTES:
            raise RuntimeError(
                f"GPU has {total / GIB:.2f} GiB total; the configured inference "
                "budget requires 28 GiB plus 2 GiB headroom"
            )
        self.total_bytes = total
        self.max_used_bytes = min(MAX_USED_BYTES, total - MIN_FREE_BYTES)
        self.pid = os.getpid()
        self._process_was_observed = False
        self._last_reported = None
        # Bound this process's PyTorch caching allocator as well as checking
        # aggregate device use, which includes other processes and CUDA pools.
        torch.cuda.set_per_process_memory_fraction(
            min(0.95, self.max_used_bytes / self.total_bytes),
            device=device_index,
        )
        _, free, _ = self.check("startup")
        if free < MIN_STARTUP_FREE_BYTES:
            raise RuntimeError(
                f"GPU startup requires at least {MIN_STARTUP_FREE_BYTES / GIB:.0f} GiB free; "
                f"only {free / GIB:.2f} GiB is available"
            )

    def process_memory_bytes(self, pid: int, *, allow_missing: bool = False) -> int:
        """Return a CUDA process allocation, failing closed when it is required."""
        process_rows = []
        for api_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            api = getattr(self._nvml, api_name, None)
            if api is None:
                continue
            try:
                process_rows.extend(api(self._handle))
            except self._nvml.NVMLError_NotSupported:
                continue
            except self._nvml.NVMLError_NoPermission:
                raise RuntimeError(
                    "NVML cannot inspect process GPU memory; refusing unmonitored inference"
                )
            except self._nvml.NVMLError:
                continue
        target_rows = [row for row in process_rows if int(getattr(row, "pid", -1)) == int(pid)]
        if not target_rows:
            if allow_missing:
                return 0
            raise RuntimeError(f"NVML has no GPU-process record for PID {pid}")
        values = [getattr(row, "usedGpuMemory", None) for row in target_rows]
        available = [int(value) for value in values if value is not None and int(value) >= 0]
        if not available:
            raise RuntimeError(
                "NVML returned no usable per-process GPU memory values; refusing unmonitored inference"
            )
        # Compute and graphics lists may report the same process; use the max
        # record instead of summing duplicate representations.
        return max(available)

    def check(self, stage: str) -> tuple[int, int, int]:
        try:
            info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            used, free = int(info.used), int(info.free)
        except Exception as exc:
            raise RuntimeError(f"NVML memory query failed at {stage}: {exc}") from exc
        if used > self.max_used_bytes or free < MIN_FREE_BYTES:
            raise RuntimeError(
                f"GPU memory safety limit reached at {stage}: "
                f"{used / GIB:.2f} GiB used, {free / GIB:.2f} GiB free; "
                f"limit {self.max_used_bytes / GIB:.2f} GiB used and "
                f"{MIN_FREE_BYTES / GIB:.2f} GiB free"
            )
        process_used = self.process_memory_bytes(
            self.pid, allow_missing=not self._process_was_observed
        )
        if process_used > 0:
            self._process_was_observed = True
        if process_used > self.max_used_bytes:
            raise RuntimeError(
                f"GPU process memory limit reached at {stage}: "
                f"PID {self.pid} uses {process_used / GIB:.2f} GiB; "
                f"limit {self.max_used_bytes / GIB:.2f} GiB"
            )
        usage = (used, free, process_used)
        if usage != self._last_reported:
            print(
                f"[gpu-memory] {stage}: {used / GIB:.2f} GiB used, "
                f"{free / GIB:.2f} GiB free, "
                f"this process {process_used / GIB:.2f} GiB"
            )
            self._last_reported = usage
        return usage
