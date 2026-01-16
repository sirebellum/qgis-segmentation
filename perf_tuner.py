"""Legacy profiling helper stub.

The plugin no longer profiles GPU/CPU settings because the runtime is now
numpy-only. This module remains as a compatibility shim for callers that still
import ``load_or_profile_settings``; it returns a default ``AdaptiveSettings``
without touching torch or running benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AdaptiveSettings:
    safety_factor: int = 1
    prefetch_depth: int = 1


def load_or_profile_settings(
    plugin_dir: str,
    device=None,
    status_callback: Optional[callable] = None,
    benchmark_runner=None,
) -> Tuple[AdaptiveSettings, bool, Optional[float]]:
    """Return a default adaptive setting without profiling."""

    if status_callback:
        try:
            status_callback("Profiling skipped: runtime is numpy-only.")
        except Exception:
            pass
    return AdaptiveSettings(), False, None


def _run_profile(device=None, status_callback=None, tile_stack=None, tiers=None):  # pragma: no cover - shim
    if status_callback:
        try:
            status_callback("Profiling disabled in numpy-only runtime.")
        except Exception:
            pass
    return AdaptiveSettings()


__all__ = ["AdaptiveSettings", "load_or_profile_settings", "_run_profile"]