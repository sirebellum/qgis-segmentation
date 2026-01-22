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
from .chunking import _compute_chunk_starts
from .common import (
    _distance_compute_dtype,
    _emit_status,
    _maybe_raise_cancel,
    _quantization_device,
    _runtime_float_dtype,
    _warn_distance_fallback,
)
from .distance import _argmin_distances_chunked, _assign_labels_tensor, _DISTANCE_CHUNK_ROWS
from .io import _materialize_raster


def _compute_kmeans_padding(resolution: int, height: int, width: int):
    height_pad = (0, (resolution - (height % resolution)) % resolution)
    width_pad = (0, (resolution - (width % resolution)) % resolution)
    return height_pad, width_pad


def _block_grid_shape(height: int, width: int, resolution: int):
    height_pad, width_pad = _compute_kmeans_padding(resolution, height, width)
    blocks_h = (height + height_pad[1]) // resolution
    blocks_w = (width + width_pad[1]) // resolution
    return blocks_h, blocks_w, height_pad, width_pad


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


def _build_label_mapping(centers: np.ndarray) -> np.ndarray:
    ordering = np.argsort(centers.mean(axis=1))
    mapping = np.zeros_like(ordering)
    mapping[ordering] = np.arange(ordering.size)
    return mapping


def _apply_label_mapping(labels_flat: np.ndarray, mapping: Optional[np.ndarray]) -> np.ndarray:
    if mapping is None:
        return labels_flat
    return mapping[labels_flat]


def _reorder_centers(centers: np.ndarray, mapping: Optional[np.ndarray]) -> np.ndarray:
    if mapping is None:
        return centers
    ordering = np.argsort(mapping)
    return centers[ordering]


def _block_chunk_plan(height: int, width: int, resolution: int, chunk_plan):
    blocks_h, blocks_w, height_pad, width_pad = _block_grid_shape(height, width, resolution)
    if chunk_plan is None:
        block_chunk = max(1, blocks_h)
    else:
        block_chunk = max(1, int(chunk_plan.chunk_size // resolution))
    block_chunk = min(block_chunk, blocks_h) if blocks_h else block_chunk
    block_chunk_w = min(block_chunk, blocks_w) if blocks_w else block_chunk
    y_starts = _compute_chunk_starts(blocks_h, block_chunk, block_chunk)
    x_starts = _compute_chunk_starts(blocks_w, block_chunk_w, block_chunk_w)
    return blocks_h, blocks_w, height_pad, width_pad, block_chunk, block_chunk_w, y_starts, x_starts


def _fit_global_centers(
    array: np.ndarray,
    num_segments: int,
    resolution: int,
    chunk_plan,
    device: torch.device,
    compute_dtype: torch.dtype,
    sample_scale: float,
    status_callback=None,
    cancel_token=None,
):
    height, width = array.shape[1], array.shape[2]
    blocks_h, blocks_w, height_pad, width_pad, block_chunk_h, block_chunk_w, y_starts, x_starts = _block_chunk_plan(
        height, width, resolution, chunk_plan
    )
    total_blocks = max(1, blocks_h * blocks_w)
    sample_scale = float(np.clip(sample_scale, 0.3, 2.0))
    base_cap = 20000
    max_samples = max(200, int(round(base_cap * sample_scale)))
    rng = np.random.default_rng(0)

    total_chunks = max(1, len(y_starts) * len(x_starts))
    remaining = max_samples
    sampled_chunks = []

    if status_callback:
        status_callback(
            f"Global K-Means sampling {min(max_samples, total_blocks)} descriptors across {total_chunks} chunk(s)."
        )

    for yb in y_starts:
        for xb in x_starts:
            _maybe_raise_cancel(cancel_token)
            yb_end = min(yb + block_chunk_h, blocks_h)
            xb_end = min(xb + block_chunk_w, blocks_w)
            y0 = yb * resolution
            x0 = xb * resolution
            y1 = min(yb_end * resolution, height)
            x1 = min(xb_end * resolution, width)
            chunk = array[:, y0:y1, x0:x1]
            descriptors, block_shape, _, _ = _smooth_and_pool_descriptors(
                chunk,
                resolution,
                device,
                compute_dtype,
                status_callback=None,
                cancel_token=cancel_token,
            )
            chunk_blocks = max(1, block_shape[0] * block_shape[1])
            target = int(round(max_samples * (chunk_blocks / total_blocks)))
            target = max(1, target)
            if remaining <= 0:
                continue
            sample_count = min(remaining, chunk_blocks, target)
            sample_idx = _sample_block_indices(block_shape, sample_count, rng)
            sampled_chunks.append(descriptors[sample_idx])
            remaining -= sample_count
            if remaining <= 0:
                break
        if remaining <= 0:
            break

    if not sampled_chunks:
        raise RuntimeError("Unable to sample descriptors for global K-Means centers.")

    sampled = np.concatenate(sampled_chunks, axis=0)
    sampled = sampled[:max_samples]
    _emit_status(
        status_callback,
        f"Fitting global K-Means centers on {sampled.shape[0]} descriptors (feature_dim={sampled.shape[1]}).",
    )
    _maybe_raise_cancel(cancel_token)
    centers = _torch_kmeans(
        sampled,
        num_clusters=num_segments,
        device=device,
        compute_dtype=compute_dtype,
        seed=0,
    )
    mapping = _build_label_mapping(centers)
    return centers, mapping, (blocks_h, blocks_w), height_pad, width_pad


def _assign_blocks_streaming(
    array: np.ndarray,
    num_segments: int,
    resolution: int,
    chunk_plan,
    centers: np.ndarray,
    mapping: Optional[np.ndarray],
    device: torch.device,
    compute_dtype: torch.dtype,
    status_callback=None,
    cancel_token=None,
):
    height, width = array.shape[1], array.shape[2]
    blocks_h, blocks_w, height_pad, width_pad, block_chunk_h, block_chunk_w, y_starts, x_starts = _block_chunk_plan(
        height, width, resolution, chunk_plan
    )
    labels_lowres = np.zeros((blocks_h, blocks_w), dtype=np.uint8)
    total = max(1, len(y_starts) * len(x_starts))
    idx = 0
    for yb in y_starts:
        for xb in x_starts:
            idx += 1
            _maybe_raise_cancel(cancel_token)
            yb_end = min(yb + block_chunk_h, blocks_h)
            xb_end = min(xb + block_chunk_w, blocks_w)
            y0 = yb * resolution
            x0 = xb * resolution
            y1 = min(yb_end * resolution, height)
            x1 = min(xb_end * resolution, width)
            if status_callback:
                status_callback(f"Assigning K-Means chunk {idx}/{total} (rows {y0}-{y1}, cols {x0}-{x1}).")
            chunk = array[:, y0:y1, x0:x1]
            descriptors, block_shape, _, _ = _smooth_and_pool_descriptors(
                chunk,
                resolution,
                device,
                compute_dtype,
                status_callback=None,
                cancel_token=cancel_token,
            )
            labels_flat = _assign_blocks_chunked(descriptors, centers, device, compute_dtype)
            labels_flat = _apply_label_mapping(labels_flat, mapping)
            labels_chunk = labels_flat.reshape(block_shape).astype(np.uint8, copy=False)
            labels_lowres[yb:yb_end, xb:xb_end] = labels_chunk[: yb_end - yb, : xb_end - xb]
    return labels_lowres, height_pad, width_pad


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


def predict_kmeans_streaming(
    array,
    num_segments=8,
    resolution=16,
    chunk_plan=None,
    status_callback=None,
    sample_scale: float = 1.0,
    return_lowres=False,
    return_centers=False,
    cancel_token=None,
    device_hint=None,
):
    _maybe_raise_cancel(cancel_token)
    array = _materialize_raster(array)
    device = _quantization_device(device_hint) or torch.device("cpu")
    compute_dtype = _distance_compute_dtype(device)
    _emit_status(
        status_callback,
        f"K-Means global centers initialized on TORCH ({num_segments} clusters, resolution={resolution}).",
    )

    centers, mapping, block_shape, height_pad, width_pad = _fit_global_centers(
        array,
        num_segments,
        resolution,
        chunk_plan,
        device,
        compute_dtype,
        sample_scale,
        status_callback=status_callback,
        cancel_token=cancel_token,
    )

    _emit_status(status_callback, "Assigning labels using global K-Means centers...")
    labels_lowres, height_pad, width_pad = _assign_blocks_streaming(
        array,
        num_segments,
        resolution,
        chunk_plan,
        centers,
        mapping,
        device,
        compute_dtype,
        status_callback=status_callback,
        cancel_token=cancel_token,
    )

    _maybe_raise_cancel(cancel_token)
    upsampled = np.repeat(np.repeat(labels_lowres, resolution, axis=0), resolution, axis=1)
    result = upsampled[: array.shape[1], : array.shape[2]].astype(np.uint8, copy=False)
    _emit_status(status_callback, "K-Means streaming output upsampled to raster resolution.")

    if not (return_lowres or return_centers):
        return result

    extras = []
    if return_lowres:
        extras.append(labels_lowres)
    if return_centers:
        extras.append(_reorder_centers(centers, mapping).astype(np.float32, copy=False))

    return (result, *extras)


def _coerce_device(device_like):
    if isinstance(device_like, torch.device):
        return device_like
    if device_like is None:
        return torch.device("cpu")
    return torch.device(device_like)


__all__ = [
    "predict_kmeans",
    "predict_kmeans_streaming",
    "_smooth_and_pool_descriptors",
    "_assign_blocks_chunked",
    "_compute_kmeans_padding",
    "_sample_block_indices",
    "_normalize_cluster_labels",
    "_build_label_mapping",
]
