# SPDX-License-Identifier: BSD-3-Clause
"""Device and memory policy helpers for training-only code.

These utilities are intentionally isolated from the QGIS runtime. They select
an appropriate torch.device and apply conservative memory limits tuned per
backend so Mac (MPS) smoke runs stay within ~50% of memory and CUDA can use up
to ~99% of VRAM. DataLoader hints are provided to keep defaults aligned with
backend capabilities.
"""
from __future__ import annotations

import os
from typing import Dict

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def select_device(preference: str | None = None) -> torch.device:
    """Pick a torch.device based on availability and a user preference.

    Args:
        preference: one of {"auto", "cuda", "mps", "cpu"} (case-insensitive).
            Defaults to "auto".
    """
    pref = (preference or "auto").strip().lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    # auto path
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_memory_policy(device: torch.device) -> None:
    """Apply backend-specific memory caps.

    - CUDA: allow up to ~99% of free VRAM using set_per_process_memory_fraction.
    - MPS: throttle PyTorch's high watermark to 50% via env
      `PYTORCH_MPS_HIGH_WATERMARK_RATIO` if the user did not override it.
    - CPU: no-op.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.99, device=device)
        except Exception:
            # Best-effort; safe to continue if unsupported on this build
            pass
        return
    if device.type == "mps":
        def _validated_ratio(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None:
                os.environ[name] = str(default)
                return default
            try:
                value = float(raw)
            except ValueError:
                value = default
            if value <= 0.0 or value > 1.0:
                value = default
                os.environ[name] = str(default)
            return value

        high = _validated_ratio("PYTORCH_MPS_HIGH_WATERMARK_RATIO", 0.5)
        low = _validated_ratio("PYTORCH_MPS_LOW_WATERMARK_RATIO", 0.4)
        if low >= high:
            low = max(0.1, min(0.9, high * 0.8))
            os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(low)
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return
    # CPU: nothing to apply


def dataloader_hints(device: torch.device, workers: int | None = None) -> Dict[str, object]:
    """Return DataLoader kwargs tuned for the selected device.

    - CUDA: enable pin_memory and persistent_workers when workers > 0.
    - MPS/CPU: leave pin_memory off by default (can be overridden).
    """
    worker_count = 0 if workers is None else max(0, int(workers))
    hints: Dict[str, object] = {
        "num_workers": worker_count,
        "pin_memory": bool(device.type == "cuda"),
        "persistent_workers": bool(device.type == "cuda" and worker_count > 0),
    }
    return hints


__all__ = ["select_device", "apply_memory_policy", "dataloader_hints"]
