# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Common runtime helpers: cancellation, device selection, dtype policies, status emitters."""
from __future__ import annotations

import warnings
from typing import Optional

import torch


class SegmentationCanceled(Exception):
    """Raised when a segmentation task is canceled mid-flight."""


def _maybe_raise_cancel(cancel_token) -> None:
    if cancel_token is None:
        return
    checker = getattr(cancel_token, "raise_if_cancelled", None)
    if callable(checker):
        checker()
        return
    probe = getattr(cancel_token, "is_cancelled", None)
    if callable(probe) and probe():
        raise SegmentationCanceled()


def _coerce_torch_device(device_like) -> Optional[torch.device]:
    if device_like is None:
        return None
    if isinstance(device_like, torch.device):
        return device_like
    if isinstance(device_like, str):
        try:
            return torch.device(device_like)
        except (TypeError, ValueError):
            return None
    return None


def _quantization_device(device_hint=None) -> Optional[torch.device]:
    candidate = _coerce_torch_device(device_hint)
    if candidate and candidate.type in {"cuda", "mps"}:
        if candidate.type == "cuda" and torch.cuda.is_available():
            return candidate
        if candidate.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return candidate
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def _runtime_float_dtype(device_hint=None) -> torch.dtype:
    """Prefer float16 tensors on GPU backends while retaining float32 on CPU."""
    device = _coerce_torch_device(device_hint)
    if device is not None and device.type in {"cuda", "mps"}:
        return torch.float16
    if device is None:
        try:
            if torch.cuda.is_available():
                return torch.float16
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.float16
        except Exception:
            pass
    return torch.float32


def _distance_compute_dtype(device_hint=None) -> torch.dtype:
    device = _coerce_torch_device(device_hint) or torch.device("cpu")
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


_distance_fallback_warned = False


def _warn_distance_fallback(reason: str) -> None:
    global _distance_fallback_warned
    if _distance_fallback_warned:
        return
    _distance_fallback_warned = True
    warnings.warn(f"Falling back to float32 distance compute ({reason}).")


def _emit_status(callback, message) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        pass
