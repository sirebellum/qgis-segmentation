# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Gaussian smoothing helpers used for stitching logits and weights."""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .common import _runtime_float_dtype
from .latent import LATENT_KNN_DEFAULTS

SMOOTH_CHUNK_BYTES = 256 * 1024 * 1024


def _gaussian_blur_channels(array, sigma, status_callback: Optional[Callable[[str], None]] = None, stage_label: str = "scores"):
    if sigma <= 0:
        return array
    return _chunked_gaussian_blur(array, sigma, status_callback=status_callback, stage_label=stage_label)


def _chunked_gaussian_blur(
    array,
    sigma,
    max_chunk_bytes=SMOOTH_CHUNK_BYTES,
    status_callback: Optional[Callable[[str], None]] = None,
    stage_label: str = "scores",
):
    sigma = max(float(sigma), 1e-6)
    channels, height, width = array.shape
    radius = int(max(1, round(3 * sigma)))
    device = _smoothing_device()
    preferred_dtype = _runtime_float_dtype(device)
    use_fp16 = preferred_dtype == torch.float16 and LATENT_KNN_DEFAULTS.get("mixed_precision_smoothing", True)
    dtype = torch.float16 if use_fp16 else torch.float32
    kernel = _build_gaussian_kernel(radius, sigma, channels, device, dtype=dtype)
    result = np.zeros_like(array, dtype=np.float32)
    bytes_per_row = max(channels * width * 4, 1)
    chunk_rows = max(radius * 2 + 1, int(max_chunk_bytes // bytes_per_row))
    chunk_rows = max(chunk_rows, radius * 2 + 1)

    start = 0
    total_chunks = max(1, math.ceil(height / chunk_rows))
    chunk_index = 0
    while start < height:
        end = min(height, start + chunk_rows)
        pad_top = min(radius, start)
        pad_bottom = min(radius, height - end)
        slice_start = start - pad_top
        slice_end = min(height, end + pad_bottom)
        chunk = array[:, slice_start:slice_end, :]
        tensor = torch.from_numpy(chunk).unsqueeze(0).to(device=device, dtype=dtype)
        padded = F.pad(tensor, (radius, radius, radius, radius), mode="reflect")
        smoothed = F.conv2d(padded, kernel, groups=channels).squeeze(0)
        usable = smoothed[:, pad_top : pad_top + (end - start), :].to(dtype)
        result[:, start:end, :] = usable.detach().cpu().numpy()
        start = end
        chunk_index += 1
        if status_callback:
            percent = int((chunk_index / total_chunks) * 100)
            status_callback(
                f"GPU smoothing [{stage_label}] chunk {chunk_index}/{total_chunks} ({percent}% complete)."
            )

    return result.astype(np.float32)


def _build_gaussian_kernel(radius, sigma, channels, device, dtype=torch.float32):
    if radius <= 0:
        kernel = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
        return kernel.repeat(channels, 1, 1, 1)
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])
    return kernel.repeat(channels, 1, 1, 1)


def _smoothing_device():
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def _build_weight_mask(size):
    if size <= 1:
        return np.ones((1, 1), dtype=np.float32)
    window = np.hanning(size)
    assert np.max(window) != 0, "np.hanning(size) returned all zeros, which should never happen for size > 1"
    mask = np.outer(window, window)
    mask = mask / np.max(mask)
    return mask.astype(np.float32)


__all__ = [
    "SMOOTH_CHUNK_BYTES",
    "_build_weight_mask",
    "_gaussian_blur_channels",
    "_chunked_gaussian_blur",
    "_build_gaussian_kernel",
    "_smoothing_device",
]
