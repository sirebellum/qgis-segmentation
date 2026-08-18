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
    return None


def _runtime_float_dtype(device_hint=None) -> torch.dtype:
    """Retain float32 on CPU."""
    return torch.float32


def _distance_compute_dtype(device_hint=None) -> torch.dtype:
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
    except Exception:  # nosec B110 - best effort status callback
        pass
