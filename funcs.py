# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Compatibility facade for runtime imports.

This module preserves the historical ``funcs`` API surface while delegating the
implementations to the ``runtime`` package. Downstream callers (tests, plugin
UI) can continue importing symbols from here without code changes.
"""
from __future__ import annotations

from typing import Any

try:
    from dependency_manager import ensure_dependencies
except ImportError:  # pragma: no cover - fallback for packaged plugin
    from .dependency_manager import ensure_dependencies  # type: ignore

ensure_dependencies()

# Common helpers
try:
    from .runtime.common import (
        SegmentationCanceled,
        _maybe_raise_cancel,
        _coerce_torch_device,
        _quantization_device,
        _runtime_float_dtype,
        _distance_compute_dtype,
        _warn_distance_fallback,
        _emit_status,
    )

    # IO helpers
    from .runtime.io import _materialize_model, _materialize_raster

    # Distance utilities
    from .runtime.distance import (
        _DISTANCE_CHUNK_ROWS,
        _prepare_reference_tensor,
        _pairwise_dist2,
        _argmin_distances_chunked,
        _topk_distances_chunked,
        _assign_labels_tensor,
        _torch_bruteforce_knn,
    )

    # Smoothing utilities
    from .runtime.smoothing import (
        _gaussian_blur_channels,
        _chunked_gaussian_blur,
        _build_gaussian_kernel,
        _smoothing_device,
        _build_weight_mask,
        SMOOTH_CHUNK_BYTES,
    )

    # Adaptive settings
    from .runtime.adaptive import (
        AdaptiveSettings,
        ChunkPlan,
        DEFAULT_MEMORY_BUDGET,
        MAX_TILE_SIZE,
        MIN_TILE_SIZE,
        VRAM_RATIO_CPU,
        VRAM_RATIO_CUDA,
        VRAM_RATIO_MPS,
        _derive_chunk_size,
        _free_vram_bytes,
        _system_available_memory,
        get_adaptive_options,
        get_adaptive_settings,
        recommended_chunk_plan,
        set_adaptive_settings,
    )

    # Chunking utilities
    from .runtime.chunking import (
        _ChunkAggregator,
        _compute_chunk_starts,
        _label_to_one_hot,
        _normalize_inference_output,
        _process_in_chunks,
    )

    # Latent refinement
    from .runtime.latent import (
        LATENT_KNN_DEFAULTS,
        _latent_knn_soft_refine,
        _resize_label_map,
        _resize_latent_map,
        _stratified_sample_indices,
    )

    # K-Means path
    from .runtime import kmeans as _runtime_kmeans
    from .runtime.kmeans import (
        predict_kmeans_streaming,
        _assign_blocks_chunked,
        _compute_kmeans_padding,
        _normalize_cluster_labels,
        _sample_block_indices,
        _smooth_and_pool_descriptors,
        _torch_kmeans,
    )

    # CNN path
    from .runtime.cnn import (
        predict_cnn,
        tile_raster,
        _apply_rotation_plan_to_volume,
        _auto_orient_tile_grid,
        _batch_to_tensor,
        _prefetch_batches,
        _prefetch_batches_cuda,
        _prefetch_batches_threaded,
        _recommended_batch_size,
    )

    # Pipeline helpers
    from .runtime.pipeline import (
        blur_segmentation_map,
        execute_cnn_segmentation,
        execute_kmeans_segmentation,
        _apply_optional_blur,
    )
except ImportError:
    from runtime.common import (
        SegmentationCanceled,
        _maybe_raise_cancel,
        _coerce_torch_device,
        _quantization_device,
        _runtime_float_dtype,
        _distance_compute_dtype,
        _warn_distance_fallback,
        _emit_status,
    )

    # IO helpers
    from runtime.io import _materialize_model, _materialize_raster

    # Distance utilities
    from runtime.distance import (
        _DISTANCE_CHUNK_ROWS,
        _prepare_reference_tensor,
        _pairwise_dist2,
        _argmin_distances_chunked,
        _topk_distances_chunked,
        _assign_labels_tensor,
        _torch_bruteforce_knn,
    )

    # Smoothing utilities
    from runtime.smoothing import (
        _gaussian_blur_channels,
        _chunked_gaussian_blur,
        _build_gaussian_kernel,
        _smoothing_device,
        _build_weight_mask,
        SMOOTH_CHUNK_BYTES,
    )

    # Adaptive settings
    from runtime.adaptive import (
        AdaptiveSettings,
        ChunkPlan,
        DEFAULT_MEMORY_BUDGET,
        MAX_TILE_SIZE,
        MIN_TILE_SIZE,
        VRAM_RATIO_CPU,
        VRAM_RATIO_CUDA,
        VRAM_RATIO_MPS,
        _derive_chunk_size,
        _free_vram_bytes,
        _system_available_memory,
        get_adaptive_options,
        get_adaptive_settings,
        recommended_chunk_plan,
        set_adaptive_settings,
    )

    # Chunking utilities
    from runtime.chunking import (
        _ChunkAggregator,
        _compute_chunk_starts,
        _label_to_one_hot,
        _normalize_inference_output,
        _process_in_chunks,
    )

    # Latent refinement
    from runtime.latent import (
        LATENT_KNN_DEFAULTS,
        _latent_knn_soft_refine,
        _resize_label_map,
        _resize_latent_map,
        _stratified_sample_indices,
    )

    # K-Means path
    import runtime.kmeans as _runtime_kmeans
    from runtime.kmeans import (
        predict_kmeans_streaming,
        _assign_blocks_chunked,
        _compute_kmeans_padding,
        _normalize_cluster_labels,
        _sample_block_indices,
        _smooth_and_pool_descriptors,
        _torch_kmeans,
    )

    # CNN path
    from runtime.cnn import (
        predict_cnn,
        tile_raster,
        _apply_rotation_plan_to_volume,
        _auto_orient_tile_grid,
        _batch_to_tensor,
        _prefetch_batches,
        _prefetch_batches_cuda,
        _prefetch_batches_threaded,
        _recommended_batch_size,
    )

    # Pipeline helpers
    from runtime.pipeline import (
        blur_segmentation_map,
        execute_cnn_segmentation,
        execute_kmeans_segmentation,
        _apply_optional_blur,
    )


def _sklearn_kmeans_fit(data_np, num_clusters: int, seed: int | None = None) -> Any:
    """Optional scikit-learn K-Means fallback retained for tests.

    The runtime defaults to the Torch path; tests may monkeypatch this symbol to
    assert the fallback is not invoked.
    """

    from sklearn.cluster import KMeans  # type: ignore

    if data_np.ndim != 2:
        raise ValueError("scikit-learn KMeans expects a 2D array")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    model = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed).fit(data_np)
    return model.cluster_centers_.astype("float32", copy=False)


def predict_kmeans(
    array,
    num_segments=8,
    resolution=16,
    status_callback=None,
    sample_scale: float = 1.0,
    return_lowres=False,
    return_centers=False,
    cancel_token=None,
    device_hint=None,
):
    """Wrapper to keep monkeypatch hooks working across the runtime split."""

    # Propagate any monkeypatches applied to this module so the runtime uses the same hooks.
    _runtime_kmeans._torch_kmeans = _torch_kmeans
    _runtime_kmeans._assign_blocks_chunked = _assign_blocks_chunked
    _runtime_kmeans._smooth_and_pool_descriptors = _smooth_and_pool_descriptors
    _runtime_kmeans._sklearn_kmeans_fit = _sklearn_kmeans_fit
    return _runtime_kmeans.predict_kmeans(
        array,
        num_segments=num_segments,
        resolution=resolution,
        status_callback=status_callback,
        sample_scale=sample_scale,
        return_lowres=return_lowres,
        return_centers=return_centers,
        cancel_token=cancel_token,
        device_hint=device_hint,
    )


__all__ = [
    # common
    "SegmentationCanceled",
    "_maybe_raise_cancel",
    "_coerce_torch_device",
    "_quantization_device",
    "_runtime_float_dtype",
    "_distance_compute_dtype",
    "_warn_distance_fallback",
    "_emit_status",
    # io
    "_materialize_model",
    "_materialize_raster",
    # distance
    "_DISTANCE_CHUNK_ROWS",
    "_prepare_reference_tensor",
    "_pairwise_dist2",
    "_argmin_distances_chunked",
    "_topk_distances_chunked",
    "_assign_labels_tensor",
    "_torch_bruteforce_knn",
    # smoothing
    "_gaussian_blur_channels",
    "_chunked_gaussian_blur",
    "_build_gaussian_kernel",
    "_smoothing_device",
    "_build_weight_mask",
    "SMOOTH_CHUNK_BYTES",
    # adaptive
    "AdaptiveSettings",
    "ChunkPlan",
    "DEFAULT_MEMORY_BUDGET",
    "MAX_TILE_SIZE",
    "MIN_TILE_SIZE",
    "VRAM_RATIO_CPU",
    "VRAM_RATIO_CUDA",
    "VRAM_RATIO_MPS",
    "_derive_chunk_size",
    "_free_vram_bytes",
    "_system_available_memory",
    "get_adaptive_options",
    "get_adaptive_settings",
    "recommended_chunk_plan",
    "set_adaptive_settings",
    # chunking
    "_ChunkAggregator",
    "_compute_chunk_starts",
    "_label_to_one_hot",
    "_normalize_inference_output",
    "_process_in_chunks",
    # latent
    "LATENT_KNN_DEFAULTS",
    "_latent_knn_soft_refine",
    "_resize_label_map",
    "_resize_latent_map",
    "_stratified_sample_indices",
    # kmeans
    "predict_kmeans",
    "predict_kmeans_streaming",
    "_assign_blocks_chunked",
    "_compute_kmeans_padding",
    "_normalize_cluster_labels",
    "_sample_block_indices",
    "_smooth_and_pool_descriptors",
    "_torch_kmeans",
    "_sklearn_kmeans_fit",
    # cnn
    "predict_cnn",
    "tile_raster",
    "_apply_rotation_plan_to_volume",
    "_auto_orient_tile_grid",
    "_batch_to_tensor",
    "_prefetch_batches",
    "_prefetch_batches_cuda",
    "_prefetch_batches_threaded",
    "_recommended_batch_size",
    # pipeline
    "blur_segmentation_map",
    "execute_cnn_segmentation",
    "execute_kmeans_segmentation",
    "_apply_optional_blur",
]
