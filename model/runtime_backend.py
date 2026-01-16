# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Backend selector for runtime inference.

Prefers torch (CUDA/MPS/CPU) when available, otherwise falls back to the numpy
runtime. This module avoids importing torch unless needed to keep the plugin
lightweight in CPU-only environments.
"""
from __future__ import annotations

import importlib
from typing import Optional

from .runtime_numpy import load_runtime_model


def _import_torch():
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def _load_torch_runtime(model_dir: str, device_preference: str, status_callback=None):
    runtime_torch = importlib.import_module("model.runtime_torch")
    return runtime_torch.load_runtime_model_torch(
        model_dir,
        device_preference=device_preference,
        status_callback=status_callback,
    )


def load_runtime(
    model_dir: str,
    *,
    prefer: str = "auto",
    device_preference: str = "auto",
    status_callback: Optional[callable] = None,
):
    """Load the best-available runtime backend.

    Args:
        model_dir: directory containing meta.json and model.npz.
        prefer: "auto" (default), "torch", or "numpy".
        device_preference: device hint for the torch backend ("auto", "cuda", "mps", "cpu").
        status_callback: optional logger for user-visible messages.
    """

    preference = (prefer or "auto").lower()
    fallback_reason = None

    if preference == "numpy":
        runtime = load_runtime_model(model_dir, status_callback=status_callback)
        runtime.backend = getattr(runtime, "backend", "numpy")
        runtime.device_label = getattr(runtime, "device_label", "cpu")
        return runtime

    torch_mod = _import_torch()
    if torch_mod is not None:
        try:
            runtime = _load_torch_runtime(model_dir, device_preference, status_callback=status_callback)
            runtime.backend = getattr(runtime, "backend", "torch")
            runtime.device_label = getattr(runtime, "device_label", str(getattr(runtime, "device", "cpu")))
            if status_callback:
                try:
                    status_callback(f"Using torch runtime on {runtime.device_label} backend.")
                except Exception:
                    pass
            return runtime
        except Exception as exc:
            fallback_reason = f"torch runtime unavailable ({exc})"
    else:
        fallback_reason = "torch is not installed"

    if status_callback and fallback_reason:
        try:
            status_callback(f"Falling back to numpy runtime: {fallback_reason}.")
        except Exception:
            pass
    runtime = load_runtime_model(model_dir, status_callback=status_callback)
    runtime.backend = getattr(runtime, "backend", "numpy")
    runtime.device_label = getattr(runtime, "device_label", "cpu")
    return runtime


__all__ = ["load_runtime"]
