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
    """Accelerated label smoothing via one-hot convolution."""
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

    # Determine device
    device = torch.device("cpu")
    dtype = torch.float32

    _emit_status(
        status_callback,
        f"Starting smoothing on CPU (kernel={kernel}px, iterations={iterations})...",
    )

    # Average kernel for box blur
    weight_single = torch.ones(
        1, 1, kernel, kernel, dtype=dtype, device=device) / float(kernel * kernel)

    # Move labels to device
    tensor = torch.from_numpy(labels_src.astype(
        np.int64, copy=False)).to(device)

    # Track the best class and its smoothed probability
    max_probs = torch.full(
        (tensor.shape[0], tensor.shape[1]), -1.0, dtype=dtype, device=device)
    best_labels = tensor.clone()

    for k in range(num_segments):
        _maybe_raise_cancel(cancel_token)

        # Create binary mask for class k
        prob_k = (tensor == k).to(dtype).unsqueeze(0).unsqueeze(0)

        # Apply convolution independently for this class
        for _ in range(iterations):
            padded = torch.nn.functional.pad(
                prob_k, (pad, pad, pad, pad), mode="replicate")
            prob_k = torch.nn.functional.conv2d(padded, weight_single)

        prob_k = prob_k.squeeze(0).squeeze(0)

        # Update max_probs and best_labels
        update_mask = prob_k > max_probs
        best_labels[update_mask] = k
        max_probs[update_mask] = prob_k[update_mask]

        _emit_status(
            status_callback,
            f"Post-smoothing segment {k + 1}/{num_segments} ({int(((k + 1) / max(num_segments, 1)) * 100)}% complete, kernel={kernel}px).",
        )

    return best_labels.cpu().numpy().astype(labels_src.dtype)


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
