# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""High-level segmentation pipelines and helpers."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .adaptive import MIN_TILE_SIZE, MAX_TILE_SIZE, recommended_chunk_plan
from .chunking import _process_in_chunks
from .common import _coerce_torch_device, _emit_status, _maybe_raise_cancel
from .io import _materialize_model, _materialize_raster
from .kmeans import predict_kmeans, predict_kmeans_streaming
from .cnn import fit_global_cnn_centers, predict_cnn, predict_cnn_with_centers

try:
    from ..autoencoder_utils import TextureAutoencoderManager
except ImportError:  # pragma: no cover
    from autoencoder_utils import TextureAutoencoderManager


def _apply_texture_autoencoder(
    raster: np.ndarray,
    labels: np.ndarray,
    texture_manager: Optional[TextureAutoencoderManager],
    status_callback,
    cancel_token,
):
    if texture_manager is None:
        return labels
    _maybe_raise_cancel(cancel_token)
    remap = texture_manager.refresh_and_remap(
        raster,
        labels,
        status_callback=status_callback,
    )
    _maybe_raise_cancel(cancel_token)
    if remap is None:
        raise RuntimeError("Texture autoencoder failed to produce a remapped result.")
    _emit_status(status_callback, "Texture autoencoder remap applied.")
    return remap


def blur_segmentation_map(
    labels: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    status_callback=None,
    cancel_token=None,
):
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
    dtype = torch.float16 if torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else torch.float32
    tensor = torch.from_numpy(labels_src.astype(np.int64, copy=False))
    one_hot = torch.nn.functional.one_hot(tensor, num_classes=num_segments).permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
    weight = torch.ones(num_segments, 1, kernel, kernel, dtype=one_hot.dtype) / float(kernel * kernel)
    result = one_hot
    for idx in range(iterations):
        _maybe_raise_cancel(cancel_token)
        padded = torch.nn.functional.pad(result, (pad, pad, pad, pad), mode="replicate")
        result = torch.nn.functional.conv2d(padded, weight, groups=result.shape[1])
        _emit_status(
            status_callback,
            f"Post-smoothing {idx + 1}/{iterations} ({int(((idx + 1) / max(iterations, 1)) * 100)}% complete, kernel={kernel}px).",
        )
    blurred = torch.argmax(result, dim=1).squeeze(0).to(tensor.dtype)
    return blurred.cpu().numpy()


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
    texture_manager: Optional[TextureAutoencoderManager] = None,
    sample_scale: float = 1.0,
    device_hint=None,
    cancel_token=None,
):
    _maybe_raise_cancel(cancel_token)
    array = _materialize_raster(array)
    device = _coerce_torch_device(device_hint) or torch.device("cpu")
    if chunk_plan is None:
        chunk_plan = recommended_chunk_plan(array.shape, device)
    if texture_manager is not None:
        set_device = getattr(texture_manager, "set_device", None)
        if callable(set_device):
            set_device(device)
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
    return _apply_texture_autoencoder(
        array,
        labels,
        texture_manager,
        status_callback,
        cancel_token,
    )


def execute_cnn_segmentation(
    cnn_model,
    array,
    num_segments,
    chunk_plan,
    tile_size,
    device,
    status_callback=None,
    profile_tier: Optional[str] = None,
    heuristic_overrides: Optional[dict] = None,
    texture_manager: Optional[TextureAutoencoderManager] = None,
    blur_config: Optional[dict] = None,
    cancel_token=None,
):
    _maybe_raise_cancel(cancel_token)
    array = _materialize_raster(array)
    device_obj = _coerce_torch_device(device) or torch.device("cpu")
    model = _materialize_model(cnn_model, device_obj)
    if chunk_plan is None:
        chunk_plan = recommended_chunk_plan(array.shape, device_obj, profile_tier=profile_tier)
    if texture_manager is not None:
        set_device = getattr(texture_manager, "set_device", None)
        if callable(set_device):
            set_device(device_obj)
    heuristic_overrides = heuristic_overrides or {}
    raw_tile_size = heuristic_overrides.get("tile_size")
    tile_size_override = int(raw_tile_size) if isinstance(raw_tile_size, (int, float)) else 0
    raw_smoothing = heuristic_overrides.get("smoothing_scale")
    smoothing_scale = float(raw_smoothing) if isinstance(raw_smoothing, (int, float)) else 1.0
    raw_latent = heuristic_overrides.get("latent_knn")
    latent_knn_config = raw_latent if isinstance(raw_latent, dict) else None

    if tile_size_override:
        tile_size = max(MIN_TILE_SIZE, min(MAX_TILE_SIZE, tile_size_override))

    height, width = array.shape[1], array.shape[2]
    effective_tile = min(tile_size, chunk_plan.chunk_size) if chunk_plan else tile_size

    def _tile_for_data(data):
        size = min(effective_tile, data.shape[1], data.shape[2])
        return max(64, size)

    if chunk_plan and chunk_plan.should_chunk(height, width):
        centers = fit_global_cnn_centers(
            model,
            array,
            num_segments,
            tile_size=_tile_for_data(array),
            device_obj=device_obj,
            chunk_plan=chunk_plan,
            status_callback=status_callback,
            memory_budget=chunk_plan.budget_bytes,
            prefetch_depth=chunk_plan.prefetch_depth,
            profile_tier=profile_tier,
            cancel_token=cancel_token,
        )
        result = _process_in_chunks(
            array,
            chunk_plan,
            num_segments,
            lambda data: predict_cnn_with_centers(
                model,
                data,
                num_segments,
                centers=centers,
                tile_size=_tile_for_data(data),
                device=device_obj,
                status_callback=status_callback,
                memory_budget=chunk_plan.budget_bytes,
                prefetch_depth=chunk_plan.prefetch_depth,
                profile_tier=profile_tier,
                latent_knn_config=latent_knn_config,
                cancel_token=cancel_token,
                return_scores=True,
            ),
            status_callback,
            smoothing_scale=smoothing_scale,
            cancel_token=cancel_token,
            harmonize_labels=False,
        )
    else:
        result = predict_cnn(
            model,
            array,
            num_segments,
            tile_size=_tile_for_data(array),
            device=device_obj,
            status_callback=status_callback,
            memory_budget=chunk_plan.budget_bytes if chunk_plan else None,
            prefetch_depth=chunk_plan.prefetch_depth if chunk_plan else None,
            profile_tier=profile_tier,
            latent_knn_config=latent_knn_config,
            cancel_token=cancel_token,
        )
    result = _apply_optional_blur(result, blur_config, status_callback, cancel_token=cancel_token)
    return _apply_texture_autoencoder(
        array,
        result,
        texture_manager,
        status_callback,
        cancel_token,
    )


__all__ = [
    "blur_segmentation_map",
    "execute_kmeans_segmentation",
    "execute_cnn_segmentation",
    "_apply_optional_blur",
]
