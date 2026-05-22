"""
Nairobi Flood Digital Twin — Memory Guard-Rail Module
=====================================================

Automated memory monitoring that enforces the safe operating envelope
defined in MEMORY_CONSTRAINTS.md:

    ┌───────────────────────────────┬──────────┬─────────────────────────┐
    │ Metric                        │ Threshold│ Action                  │
    ├───────────────────────────────┼──────────┼─────────────────────────┤
    │ Process RSS                   │ ≤ 12 GB  │ Log WARNING             │
    │ Process RSS                   │ > 14 GB  │ Log CRITICAL, abort     │
    │ Single tensor allocation      │ > 2 GB   │ Raise MemoryError       │
    │ PyTorch reserved memory       │ > 8 GB   │ Log WARNING, trigger GC │
    └───────────────────────────────┴──────────┴─────────────────────────┘

Usage
-----
    from src.utils.memory_check import MemoryGuard

    guard = MemoryGuard()          # runs startup check automatically
    guard.check_rss()              # poll process RSS
    guard.check_tensor_alloc(t)    # validate a single tensor
    guard.epoch_cleanup()          # post-epoch GC sweep
"""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import psutil
from loguru import logger

# ---------------------------------------------------------------------------
#  Constants — sourced from MEMORY_CONSTRAINTS.md
# ---------------------------------------------------------------------------

_GB = 1 << 30  # 1 GiB in bytes

SYSTEM_RAM_GB: int = 16
MIN_AVAILABLE_GB: float = 4.0       # abort on boot if less
RSS_WARN_GB: float = 12.0           # WARNING threshold
RSS_CRITICAL_GB: float = 14.0       # CRITICAL threshold — abort batch
TENSOR_MAX_GB: float = 2.0          # single-tensor hard cap
TORCH_RESERVED_WARN_GB: float = 8.0 # PyTorch memory-pool WARNING


# ---------------------------------------------------------------------------
#  Structured log format
# ---------------------------------------------------------------------------

_MEMORY_LOG_FMT = (
    "{time:YYYY-MM-DDTHH:mm:ss} | {level:<8} | "
    "{extra[metric_name]} | "
    "value_gb={extra[value_gb]:.3f} | "
    "threshold_gb={extra[threshold_gb]:.3f} | "
    "action={extra[action_taken]}"
)


def _configure_memory_logger() -> int:
    """Add a dedicated memory-event sink to loguru and return its id."""
    return logger.add(
        sys.stderr,
        format=_MEMORY_LOG_FMT,
        filter=lambda record: "metric_name" in record["extra"],
        level="WARNING",
    )


# ---------------------------------------------------------------------------
#  Helper — safe PyTorch import (torch is optional at import time)
# ---------------------------------------------------------------------------

def _try_import_torch() -> Any:
    """Return the ``torch`` module or ``None`` if unavailable."""
    try:
        import torch  # noqa: F811
        return torch
    except ImportError:
        return None


# ---------------------------------------------------------------------------
#  MemoryGuard
# ---------------------------------------------------------------------------

@dataclass
class MemoryGuard:
    """
    Centralised memory watchdog.

    Instantiate once at process startup; call check methods at critical
    junctures (training loop iterations, data-load batches, etc.).
    """

    rss_warn_gb: float = RSS_WARN_GB
    rss_critical_gb: float = RSS_CRITICAL_GB
    tensor_max_gb: float = TENSOR_MAX_GB
    torch_reserved_warn_gb: float = TORCH_RESERVED_WARN_GB
    _logger_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger_id = _configure_memory_logger()
        self._startup_check()

    # ------------------------------------------------------------------
    #  Startup check
    # ------------------------------------------------------------------

    def _startup_check(self) -> None:
        """Abort if available physical memory is below the safe floor."""
        mem = psutil.virtual_memory()
        available_gb = mem.available / _GB

        if available_gb < MIN_AVAILABLE_GB:
            self._log(
                "CRITICAL",
                metric_name="startup_available_ram",
                value_gb=available_gb,
                threshold_gb=MIN_AVAILABLE_GB,
                action_taken="ABORT — insufficient RAM",
            )
            sys.exit(
                f"[MemoryGuard] ABORT: only {available_gb:.1f} GB available "
                f"(minimum {MIN_AVAILABLE_GB:.1f} GB required)."
            )

        self._log(
            "INFO",
            metric_name="startup_available_ram",
            value_gb=available_gb,
            threshold_gb=MIN_AVAILABLE_GB,
            action_taken="OK",
        )

    # ------------------------------------------------------------------
    #  RSS polling
    # ------------------------------------------------------------------

    def check_rss(self) -> float:
        """
        Sample current process RSS.

        Returns the RSS in GB.  Logs WARNING at ``rss_warn_gb`` and
        CRITICAL (+ raises ``MemoryError``) at ``rss_critical_gb``.
        """
        process = psutil.Process(os.getpid())
        rss_bytes = process.memory_info().rss
        rss_gb = rss_bytes / _GB

        if rss_gb > self.rss_critical_gb:
            self._log(
                "CRITICAL",
                metric_name="process_rss",
                value_gb=rss_gb,
                threshold_gb=self.rss_critical_gb,
                action_taken="ABORT_BATCH",
            )
            raise MemoryError(
                f"[MemoryGuard] Process RSS ({rss_gb:.2f} GB) exceeds "
                f"critical threshold ({self.rss_critical_gb} GB). "
                "Aborting current batch."
            )

        if rss_gb > self.rss_warn_gb:
            self._log(
                "WARNING",
                metric_name="process_rss",
                value_gb=rss_gb,
                threshold_gb=self.rss_warn_gb,
                action_taken="WARN_CONTINUE",
            )

        return rss_gb

    # ------------------------------------------------------------------
    #  Single-tensor guard
    # ------------------------------------------------------------------

    def check_tensor_alloc(self, tensor: Any) -> None:
        """
        Validate that a single tensor does not exceed the allocation cap.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to inspect.

        Raises
        ------
        MemoryError
            If the tensor's storage exceeds ``tensor_max_gb``.
        """
        torch = _try_import_torch()
        if torch is None or not isinstance(tensor, torch.Tensor):
            return

        nbytes = tensor.nelement() * tensor.element_size()
        size_gb = nbytes / _GB

        if size_gb > self.tensor_max_gb:
            self._log(
                "CRITICAL",
                metric_name="single_tensor_alloc",
                value_gb=size_gb,
                threshold_gb=self.tensor_max_gb,
                action_taken="REJECT_ALLOC",
            )
            raise MemoryError(
                f"[MemoryGuard] Tensor allocation ({size_gb:.3f} GB) exceeds "
                f"the {self.tensor_max_gb} GB single-tensor cap."
            )

    # ------------------------------------------------------------------
    #  PyTorch reserved-memory check
    # ------------------------------------------------------------------

    def check_torch_reserved(self) -> float | None:
        """
        Check PyTorch's memory-pool reservation (CUDA or CPU fallback).

        Returns reserved memory in GB, or ``None`` if torch is unavailable
        or running on CPU (where reserved-memory tracking is a no-op).
        """
        torch = _try_import_torch()
        if torch is None:
            return None

        if not torch.cuda.is_available():
            # CPU-only: nothing to track — always safe
            return 0.0

        reserved_bytes = torch.cuda.memory_reserved()
        reserved_gb = reserved_bytes / _GB

        if reserved_gb > self.torch_reserved_warn_gb:
            self._log(
                "WARNING",
                metric_name="torch_reserved_memory",
                value_gb=reserved_gb,
                threshold_gb=self.torch_reserved_warn_gb,
                action_taken="TRIGGER_GC",
            )
            gc.collect()
            torch.cuda.empty_cache()

        return reserved_gb

    # ------------------------------------------------------------------
    #  Epoch-end cleanup
    # ------------------------------------------------------------------

    def epoch_cleanup(self) -> None:
        """
        Force garbage collection and clear PyTorch cache.

        Call at the end of every training / inference epoch to reclaim
        unreferenced memory before the next batch window opens.
        """
        gc.collect()

        torch = _try_import_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("[MemoryGuard] Epoch cleanup complete — GC + cache cleared.")

    # ------------------------------------------------------------------
    #  Full diagnostic sweep
    # ------------------------------------------------------------------

    def run_diagnostics(self) -> dict[str, float | None]:
        """
        Execute all checks and return a snapshot dict.

        Returns
        -------
        dict
            Keys: ``rss_gb``, ``torch_reserved_gb``, ``available_gb``.
        """
        rss_gb = self.check_rss()
        torch_reserved_gb = self.check_torch_reserved()
        available_gb = psutil.virtual_memory().available / _GB

        return {
            "rss_gb": rss_gb,
            "torch_reserved_gb": torch_reserved_gb,
            "available_gb": available_gb,
        }

    # ------------------------------------------------------------------
    #  Batch-size auto-scaling helper
    # ------------------------------------------------------------------

    @staticmethod
    def recommend_batch_size(current_batch_size: int) -> int:
        """
        Halve the batch size if process RSS is above the warning line.

        Parameters
        ----------
        current_batch_size : int
            Current batch size in use.

        Returns
        -------
        int
            Adjusted batch size (minimum 1).
        """
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / _GB

        if rss_gb > RSS_WARN_GB:
            new_size = max(1, current_batch_size // 2)
            logger.warning(
                f"[MemoryGuard] RSS={rss_gb:.2f} GB > {RSS_WARN_GB} GB — "
                f"halving batch size: {current_batch_size} → {new_size}"
            )
            return new_size

        return current_batch_size

    # ------------------------------------------------------------------
    #  Internal logging helper
    # ------------------------------------------------------------------

    @staticmethod
    def _log(
        level: str,
        *,
        metric_name: str,
        value_gb: float,
        threshold_gb: float,
        action_taken: str,
    ) -> None:
        """Emit a structured memory-event log line via loguru."""
        logger.bind(
            metric_name=metric_name,
            value_gb=value_gb,
            threshold_gb=threshold_gb,
            action_taken=action_taken,
        ).log(level, "Memory event: {}", action_taken)


# ---------------------------------------------------------------------------
#  Module-level convenience — run a quick diagnostic when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Nairobi Flood Digital Twin — Memory Diagnostics")
    print("=" * 60)

    guard = MemoryGuard()
    diagnostics = guard.run_diagnostics()

    for key, val in diagnostics.items():
        display = f"{val:.3f} GB" if val is not None else "N/A"
        print(f"  {key:.<30s} {display}")

    print()
    print("Thresholds:")
    print(f"  RSS WARNING ................ {RSS_WARN_GB} GB")
    print(f"  RSS CRITICAL ............... {RSS_CRITICAL_GB} GB")
    print(f"  Single tensor cap .......... {TENSOR_MAX_GB} GB")
    print(f"  Torch reserved WARNING ..... {TORCH_RESERVED_WARN_GB} GB")
    print("=" * 60)
