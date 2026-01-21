# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Torch-based K-Means segmentation utilities."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .adaptive import _derive_chunk_size
from .common import _distance_compute_dtype, _emit_status, _maybe_raise_cancel, _quantization_device, _runtime_float_dtype, _warn_distance_fallback
from .distance import _argmin_distances_chunked, _assign_labels_tensor, _DISTANCE_CHUNK_ROWS
from .io import _materialize_raster


def _compute_kmeans_padding(resolution: int, height: int, width: int):
    height_pad = (0, (resolution - (height % resolution)) % resolution)
    width_pad = (0, (resolution - (width % resolution)) % resolution)
    return height_pad, width_pad


def _smooth_and_pool_descriptors(
    array: np.ndarray,
    resolution: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    status_callback=None,
    cancel_token=None,
):
    _maybe_raise_cancel(cancel_token)
    height_pad, width_pad = _compute_kmeans_padding(resolution, array.shape[1], array.shape[2])
    channel_pad = (0, 0)
    array_padded = np.pad(array, (channel_pad, height_pad, width_pad), mode="constant")
    _emit_status(status_callback, "Raster padded for block processing.")

    tensor = torch.as_tensor(array_padded, device=device, dtype=torch.float32)
    target_dtype = compute_dtype if compute_dtype == torch.float16 and device.type != "cpu" else torch.float32
    try:
        smoothed = F.avg_pool2d(tensor.to(dtype=target_dtype).unsqueeze(0), kernel_size=3, stride=1, padding=1)
    except RuntimeError:
        _warn_distance_fallback("fp16 smoothing instability")
        smoothed = F.avg_pool2d(tensor.to(dtype=torch.float32).unsqueeze(0), kernel_size=3, stride=1, padding=1)
    pooled = F.avg_pool2d(smoothed, kernel_size=resolution, stride=resolution)

    pooled = pooled.squeeze(0).permute(1, 2, 0).contiguous()
    pooled = pooled.to(torch.float32)
    blocks_h, blocks_w, channels = pooled.shape
    descriptors = pooled.view(-1, channels).cpu().numpy()
    _emit_status(
        status_callback,
        f"Smoothed features and pooled {blocks_h * blocks_w} blocks ({channels} channels, resolution={resolution}).",
    )
    return descriptors, (blocks_h, blocks_w), height_pad, width_pad


def _sample_block_indices(block_shape, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    blocks_h, blocks_w = block_shape
    total = blocks_h * blocks_w
    if total <= max_samples:
        return np.arange(total, dtype=np.int64)
    stride = max(1, int(math.sqrt(total / max_samples)))
    coords = [(y, x) for y in range(0, blocks_h, stride) for x in range(0, blocks_w, stride)]
    if len(coords) > max_samples:
        chosen = rng.choice(len(coords), size=max_samples, replace=False)
        coords = [coords[i] for i in chosen]
    else:
        rng.shuffle(coords)
        coords = coords[:max_samples]
    return np.array([y * blocks_w + x for y, x in coords], dtype=np.int64)


def _assign_blocks_chunked(
    descriptors: np.ndarray,
    centers: np.ndarray,
    device: torch.device,
    compute_dtype: torch.dtype,
    chunk_rows: int = _DISTANCE_CHUNK_ROWS,
):
    return _argmin_distances_chunked(
        descriptors,
        torch.as_tensor(centers),
        device,
        compute_dtype,
        chunk_rows=chunk_rows,
    )


def _torch_kmeans(
    data_np: np.ndarray,
    num_clusters: int,
    device: Optional[torch.device],
    compute_dtype: torch.dtype,
    max_iters: int = 25,
    tol: float = 1e-4,
    seed: Optional[int] = None,
):
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if data_np.size == 0:
        raise ValueError("Input data for K-Means must not be empty")
    device_obj = _coerce_device(device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device_obj)
        generator.manual_seed(int(seed))

    data_tensor = torch.as_tensor(data_np, device=device_obj, dtype=torch.float32)
    n_samples, feature_dim = data_tensor.shape
    if n_samples < num_clusters:
        raise ValueError("num_clusters cannot exceed the number of samples")

    init_idx = torch.randperm(n_samples, generator=generator, device=device_obj)[:num_clusters]
    centers_fp32 = data_tensor[init_idx].float().clone()
    current_dtype = compute_dtype

    for _ in range(max_iters):
        labels = _assign_labels_tensor(data_tensor, centers_fp32, current_dtype)
        counts = torch.bincount(labels, minlength=num_clusters)
        sums = torch.zeros((num_clusters, feature_dim), device=device_obj, dtype=torch.float32)
        sums.index_add_(0, labels, data_tensor)
        empty = counts == 0
        if empty.any():
            refill = torch.randperm(n_samples, generator=generator, device=device_obj)[: int(empty.sum())]
            sums[empty] = data_tensor[refill]
            counts[empty] = 1
        counts = counts.to(torch.float32)
        new_centers = sums / counts.unsqueeze(1)
        shift = torch.norm(centers_fp32 - new_centers, dim=1).max().item()
        centers_fp32 = new_centers
        if shift <= tol:
            break
        current_dtype = compute_dtype

    return centers_fp32.cpu().numpy().astype(np.float32, copy=False)


def _normalize_cluster_labels(labels, centers):
    n_centers = centers.shape[0]
    flat = labels.reshape(-1)
    if not np.all((flat >= 0) & (flat < n_centers)):
        raise ValueError(
            f"All label indices must be in [0, {n_centers-1}]. Found out-of-bounds values: {flat[(flat < 0) | (flat >= n_centers)]}"
        )
    ordering = np.argsort(centers.mean(axis=1))
    mapping = np.zeros_like(ordering)
    mapping[ordering] = np.arange(ordering.size)
    flat = mapping[flat]
    return flat.reshape(labels.shape)


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
    _maybe_raise_cancel(cancel_token)
    device = _quantization_device(device_hint) or torch.device("cpu")
    compute_dtype = _distance_compute_dtype(device)
    _emit_status(
        status_callback,
        f"K-Means initialized on TORCH ({num_segments} clusters, resolution={resolution}).",
    )

    descriptors, block_shape, _, _ = _smooth_and_pool_descriptors(
        array,
        resolution,
        device,
        compute_dtype,
        status_callback=status_callback,
        cancel_token=cancel_token,
    )

    blocks_h, blocks_w = block_shape
    rng = np.random.default_rng()
    sample_scale = float(np.clip(sample_scale, 0.3, 2.0))
    base_cap = 20000
    max_samples = max(200, int(round(base_cap * sample_scale)))
    sample_idx = _sample_block_indices(block_shape, max_samples, rng)
    sampled = descriptors[sample_idx]
    _emit_status(
        status_callback,
        f"Fitting Torch K-Means on {sampled.shape[0]} pooled descriptors (feature_dim={sampled.shape[1]}).",
    )

    _maybe_raise_cancel(cancel_token)
    centers = _torch_kmeans(
        sampled,
        num_clusters=num_segments,
        device=device,
        compute_dtype=compute_dtype,
        seed=0,
    )

    _maybe_raise_cancel(cancel_token)
    _emit_status(status_callback, f"Assigning clusters on TORCH backend over {descriptors.shape[0]} blocks...")
    labels_flat = _assign_blocks_chunked(descriptors, centers, device, compute_dtype)
    labels_flat = _normalize_cluster_labels(labels_flat, centers)
    _emit_status(status_callback, "Cluster assignment complete.")

    labels_lowres = labels_flat.reshape(blocks_h, blocks_w).astype(np.uint8, copy=False)

    _maybe_raise_cancel(cancel_token)
    upsampled = np.repeat(np.repeat(labels_lowres, resolution, axis=0), resolution, axis=1)
    result = upsampled[: array.shape[1], : array.shape[2]].astype(np.uint8, copy=False)
    _emit_status(status_callback, "K-Means output upsampled to raster resolution.")

    if not (return_lowres or return_centers):
        return result

    extras = []
    if return_lowres:
        extras.append(labels_lowres)
    if return_centers:
        extras.append(centers.astype(np.float32, copy=False))

    return (result, *extras)


def _coerce_device(device_like):
    if isinstance(device_like, torch.device):
        return device_like
    if device_like is None:
        return torch.device("cpu")
    return torch.device(device_like)


__all__ = [
    "predict_kmeans",
    "_smooth_and_pool_descriptors",
    "_assign_blocks_chunked",
    "_compute_kmeans_padding",
    "_sample_block_indices",
    "_normalize_cluster_labels",
]
