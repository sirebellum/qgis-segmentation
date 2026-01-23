# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""High-level segmentation pipelines and helpers."""
from __future__ import annotations

import numpy as np
import torch

from .adaptive import recommended_chunk_plan
from .common import _coerce_torch_device, _emit_status, _maybe_raise_cancel
from .io import _materialize_raster
from .kmeans import predict_kmeans, predict_kmeans_streaming
from .smoothing import _smoothing_device


def blur_segmentation_map(
    labels: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    status_callback=None,
    cancel_token=None,
):
    """GPU-accelerated label smoothing via one-hot convolution."""
    if labels is None or labels.size == 0 or labels.ndim != 2:
        return labels
    kernel = int(max(1, kernel_size))
    iterations = int(max(1, iterations))
    if kernel <= 1 or iterations <= 0:
        return labels
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    labels_src = np.ascontiguousarray(labels)
    num_segments = int(labels_src.max()) + 1
    if num_segments <= 0:
        return labels
    _maybe_raise_cancel(cancel_token)

    # Determine device for GPU acceleration
    device = _smoothing_device()
    use_gpu = device.type in ("cuda", "mps")
    dtype = torch.float16 if use_gpu else torch.float32

    _emit_status(
        status_callback,
        f"Starting GPU smoothing on {device.type.upper()} (kernel={kernel}px, iterations={iterations})...",
    )

    # Build one-hot encoding on GPU
    tensor = torch.from_numpy(labels_src.astype(np.int64, copy=False)).to(device)
    one_hot = torch.nn.functional.one_hot(tensor, num_classes=num_segments)
    one_hot = one_hot.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype, device=device)

    # Averaging kernel for box blur
    weight = torch.ones(num_segments, 1, kernel, kernel, dtype=dtype, device=device) / float(kernel * kernel)

    result = one_hot
    for idx in range(iterations):
        _maybe_raise_cancel(cancel_token)
        padded = torch.nn.functional.pad(result, (pad, pad, pad, pad), mode="replicate")
        result = torch.nn.functional.conv2d(padded, weight, groups=result.shape[1])
        _emit_status(
            status_callback,
            f"Post-smoothing {idx + 1}/{iterations} ({int(((idx + 1) / max(iterations, 1)) * 100)}% complete, kernel={kernel}px).",
        )

    blurred = torch.argmax(result, dim=1).squeeze(0).to(torch.int64)
    return blurred.cpu().numpy().astype(labels_src.dtype)


def _apply_optional_blur(labels, blur_config, status_callback, cancel_token=None):
    if blur_config is None:
        return labels
    kernel = int(max(1, blur_config.get("kernel_size", 1) or 1))
    iterations = int(max(1, blur_config.get("iterations", 1) or 1))
    if kernel <= 1 or iterations <= 0:
        return labels
    return blur_segmentation_map(
        labels,
        kernel,
        iterations,
        status_callback=status_callback,
        cancel_token=cancel_token,
    )


def execute_kmeans_segmentation(
    array,
    num_segments,
    resolution,
    chunk_plan,
    status_callback=None,
    sample_scale: float = 1.0,
    device_hint=None,
    cancel_token=None,
):
    _maybe_raise_cancel(cancel_token)
    array = _materialize_raster(array)
    device = _coerce_torch_device(device_hint) or torch.device("cpu")
    if chunk_plan is None:
        chunk_plan = recommended_chunk_plan(array.shape, device)
    height, width = array.shape[1], array.shape[2]
    if chunk_plan and chunk_plan.should_chunk(height, width):
        labels = predict_kmeans_streaming(
            array,
            num_segments,
            resolution,
            chunk_plan=chunk_plan,
            status_callback=status_callback,
            sample_scale=sample_scale,
            cancel_token=cancel_token,
            device_hint=device,
        )
    else:
        labels = predict_kmeans(
            array,
            num_segments,
            resolution,
            status_callback=status_callback,
            sample_scale=sample_scale,
            cancel_token=cancel_token,
            device_hint=device,
        )
    return labels


__all__ = [
    "blur_segmentation_map",
    "execute_kmeans_segmentation",
    "_apply_optional_blur",
]
