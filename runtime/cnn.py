# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""TorchScript CNN inference path with tiling and latent refinement."""
from __future__ import annotations

import math
from collections import deque
from contextlib import nullcontext
from typing import Iterable, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F

from .adaptive import AdaptiveSettings, DEFAULT_MEMORY_BUDGET, MIN_TILE_SIZE, MAX_TILE_SIZE, get_adaptive_settings
from .chunking import _compute_chunk_starts, _label_to_one_hot
from .common import _emit_status, _maybe_raise_cancel, _quantization_device, _runtime_float_dtype
from .distance import _DISTANCE_CHUNK_ROWS
from .io import _materialize_model, _materialize_raster
from .latent import _latent_knn_soft_refine
from .kmeans import (
    _assign_blocks_chunked,
    _build_label_mapping,
    _reorder_centers,
    _sample_block_indices,
    _torch_kmeans,
)


def tile_raster(array, tile_size):
    padding = lambda shape: 0 if shape % tile_size == 0 else tile_size - shape % tile_size
    channel_pad = (0, 0)
    height_pad = (0, padding(array.shape[1]))
    width_pad = (0, padding(array.shape[2]))
    array_padded = np.pad(
        array,
        (channel_pad, height_pad, width_pad),
        mode="constant",
    )

    tiles = array_padded.reshape(
        array_padded.shape[0],
        array_padded.shape[1] // tile_size,
        tile_size,
        array_padded.shape[2] // tile_size,
        tile_size,
    )
    tiles = tiles.transpose(1, 3, 0, 2, 4)
    tiles = tiles.reshape(
        tiles.shape[0] * tiles.shape[1],
        array_padded.shape[0],
        tile_size,
        tile_size,
    )

    grid_shape = (
        array_padded.shape[1] // tile_size,
        array_padded.shape[2] // tile_size,
    )

    return tiles, (height_pad[1], width_pad[1]), grid_shape


ROTATION_ALIGNMENT_THRESHOLD = 0.6
ROTATION_IMPROVEMENT_MARGIN = 0.05


def _auto_orient_tile_grid(label_map: np.ndarray, tile_size: int):
    if tile_size <= 1:
        return label_map, None
    height, width = label_map.shape
    rows = max(1, math.ceil(height / tile_size))
    cols = max(1, math.ceil(width / tile_size))
    oriented = np.zeros_like(label_map)
    rotation_plan = np.zeros((rows, cols), dtype=np.uint8)
    changed = False

    for r in range(rows):
        y0 = r * tile_size
        if y0 >= height:
            break
        y1 = min(y0 + tile_size, height)
        for c in range(cols):
            x0 = c * tile_size
            if x0 >= width:
                break
            x1 = min(x0 + tile_size, width)
            tile = label_map[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            if tile.shape[0] != tile.shape[1]:
                oriented[y0:y1, x0:x1] = tile
                rotation_plan[r, c] = 0
                continue
            top_edge = oriented[y0 - 1, x0:x1] if y0 > 0 else None
            left_edge = oriented[y0:y1, x0 - 1] if x0 > 0 else None
            best_k, best_tile = _select_tile_rotation(tile, top_edge, left_edge)
            rotation_plan[r, c] = best_k
            if best_k != 0:
                changed = True
            oriented[y0:y1, x0:x1] = best_tile

    if not changed:
        return label_map, None
    return oriented, rotation_plan


def _select_tile_rotation(tile: np.ndarray, top_edge: Optional[np.ndarray], left_edge: Optional[np.ndarray]):
    best_k = 0
    best_score = float("-inf")
    best_tile = tile
    base_score = None
    for k in range(4):
        candidate = np.rot90(tile, k=k)
        score = 0.0
        if top_edge is not None and top_edge.size > 0:
            overlap = min(candidate.shape[1], top_edge.shape[-1])
            if overlap > 0:
                score += float(np.mean(candidate[0, :overlap] == top_edge[:overlap]))
        if left_edge is not None and left_edge.size > 0:
            overlap = min(candidate.shape[0], left_edge.shape[0])
            if overlap > 0:
                score += float(np.mean(candidate[:overlap, 0] == left_edge[:overlap]))
        if k == 0:
            base_score = score
        if score > best_score + 1e-6:
            best_score = score
            best_k = k
            best_tile = candidate
    if best_k != 0:
        baseline = 0.0 if base_score is None else base_score
        if best_score < ROTATION_ALIGNMENT_THRESHOLD:
            return 0, tile
        if (best_score - baseline) < ROTATION_IMPROVEMENT_MARGIN:
            return 0, tile
    return best_k, best_tile


def _apply_rotation_plan_to_volume(volume: np.ndarray, tile_size: int, plan: np.ndarray):
    rotated = np.zeros_like(volume)
    channels, height, width = volume.shape
    rows, cols = plan.shape
    for r in range(rows):
        y0 = r * tile_size
        if y0 >= height:
            break
        y1 = min(y0 + tile_size, height)
        for c in range(cols):
            x0 = c * tile_size
            if x0 >= width:
                break
            x1 = min(x0 + tile_size, width)
            tile = volume[:, y0:y1, x0:x1]
            if tile.size == 0:
                continue
            if tile.shape[1] != tile.shape[2]:
                rotated[:, y0:y1, x0:x1] = tile
                continue
            k = int(plan[r, c]) if plan is not None else 0
            if k == 0:
                rotated[:, y0:y1, x0:x1] = tile
            else:
                rotated[:, y0:y1, x0:x1] = np.rot90(tile, k=k, axes=(-2, -1))
    return rotated


def _recommended_batch_size(channels, height, width, memory_budget, settings: AdaptiveSettings):
    channels = max(1, channels)
    height = max(1, height)
    width = max(1, width)
    bytes_per_tile = channels * height * width * 4
    budget = max(memory_budget or DEFAULT_MEMORY_BUDGET, bytes_per_tile)
    safety = max(1, settings.safety_factor)
    depth = max(1, settings.prefetch_depth)
    effective_budget = max(1, budget // safety)
    denom = max(bytes_per_tile * (1 + depth), 1)
    return max(1, effective_budget // denom)


def _prefetch_batches(tiles, batch_size, device, depth=2, cancel_token=None, dtype=None):
    total = tiles.shape[0]
    if total == 0:
        return
    target_dtype = dtype or _runtime_float_dtype(device)
    if device.type == "cuda" and torch.cuda.is_available():
        yield from _prefetch_batches_cuda(tiles, batch_size, device, cancel_token, dtype=target_dtype)
        return
    yield from _prefetch_batches_threaded(tiles, batch_size, device, depth, cancel_token, dtype=target_dtype)


def _prefetch_batches_threaded(tiles, batch_size, device, depth=2, cancel_token=None, dtype=None):
    from concurrent.futures import ThreadPoolExecutor

    total = tiles.shape[0]
    depth = max(1, depth or 1)
    executor = ThreadPoolExecutor(max_workers=depth)
    futures = deque()
    index = 0

    try:
        while index < total or futures:
            while index < total and len(futures) < depth:
                _maybe_raise_cancel(cancel_token)
                start = index
                end = min(start + batch_size, total)
                batch = tiles[start:end]
                future = executor.submit(_batch_to_tensor, batch, device, dtype)
                futures.append((future, start, end))
                index = end
            future, start, end = futures.popleft()
            try:
                _maybe_raise_cancel(cancel_token)
                yield future.result(), start, end, total
            except Exception:
                while futures:
                    leftover_future, _, _ = futures.popleft()
                    try:
                        leftover_future.result()
                    except Exception:
                        pass
                raise
    finally:
        executor.shutdown(wait=True)


def _prefetch_batches_cuda(tiles, batch_size, device, cancel_token=None, dtype=None):
    total = tiles.shape[0]
    if total == 0:
        return
    stream = torch.cuda.Stream(device=device)
    next_start = 0
    next_tensor = None
    next_bounds = None

    def _stage_copy(start, end):
        _maybe_raise_cancel(cancel_token)
        batch = torch.from_numpy(tiles[start:end])
        if not batch.is_floating_point():
            batch = batch.float().div_(255.0)
        else:
            batch = batch / 255.0
        if dtype is not None and batch.dtype != dtype:
            batch = batch.to(dtype)
        pinned = batch.pin_memory()
        with torch.cuda.stream(stream):
            gpu_tensor = pinned.to(device, non_blocking=True)
        del pinned
        return gpu_tensor

    if next_start < total:
        staged_end = min(next_start + batch_size, total)
        next_tensor = _stage_copy(next_start, staged_end)
        next_bounds = (next_start, staged_end)
        next_start = staged_end

    current_stream = torch.cuda.current_stream(device)
    while next_tensor is not None:
        current_stream.wait_stream(stream)
        tensor = next_tensor
        if next_bounds is None:
            break
        start, end = next_bounds
        if next_start < total:
            staged_end = min(next_start + batch_size, total)
            next_tensor = _stage_copy(next_start, staged_end)
            next_bounds = (next_start, staged_end)
            next_start = staged_end
        else:
            next_tensor = None
            next_bounds = None
        yield tensor, start, end, total


def _batch_to_tensor(batch, device, dtype):
    tensor = torch.from_numpy(batch)
    if not tensor.is_floating_point():
        tensor = tensor.float().div_(255.0)
    else:
        tensor = tensor / 255.0
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor.to(device)


def predict_cnn(
    cnn_model,
    array,
    num_segments,
    tile_size=256,
    device="cpu",
    status_callback=None,
    memory_budget=None,
    prefetch_depth=None,
    profile_tier: Optional[str] = None,
    latent_knn_config: Optional[dict] = None,
    cancel_token=None,
    return_scores=False,
    centers: Optional[np.ndarray] = None,
    require_centers: bool = False,
):
    _maybe_raise_cancel(cancel_token)
    device_obj = cast(torch.device, _quantization_device(device) or torch.device("cpu"))
    compute_dtype = _runtime_float_dtype(device_obj)
    effective_budget = memory_budget or DEFAULT_MEMORY_BUDGET
    settings = get_adaptive_settings(effective_budget, tier=profile_tier)
    prefetch_depth = prefetch_depth or settings.prefetch_depth
    memory_budget = effective_budget

    if require_centers and centers is None:
        raise RuntimeError("Global centers are required for CNN chunk assignment.")

    coverage_map, height_pad, width_pad, grid_shape = _compute_latent_grid(
        cnn_model,
        array,
        tile_size,
        device_obj,
        status_callback=status_callback,
        memory_budget=memory_budget,
        prefetch_depth=prefetch_depth,
        profile_tier=profile_tier,
        cancel_token=cancel_token,
    )

    latent_grid = coverage_map.copy()
    lowres_scores = None
    used_global_centers = bool(require_centers and centers is not None)
    if centers is None:
        from .kmeans import predict_kmeans  # local import to avoid cycle

        kmeans_outputs = predict_kmeans(
            coverage_map,
            num_segments=num_segments,
            resolution=1,
            status_callback=status_callback,
            return_lowres=True,
            return_centers=True,
            cancel_token=cancel_token,
            device_hint=device_obj,
        )
        if isinstance(kmeans_outputs, tuple):
            _, lowres_labels, centers = kmeans_outputs
        else:
            lowres_labels = None
            centers = None
    else:
        lowres_labels, lowres_scores = _assign_latent_labels(
            latent_grid,
            centers,
            num_segments,
            device_obj,
            compute_dtype,
            status_callback=status_callback,
            latent_knn_config=latent_knn_config,
            cancel_token=cancel_token,
            return_scores=return_scores,
        )
        kmeans_outputs = (None, lowres_labels, centers)

    if centers is not None and lowres_labels is not None:
        if used_global_centers:
            refined_lowres = lowres_labels
        else:
            refined_output = _latent_knn_soft_refine(
                latent_grid,
                lowres_labels,
                centers,
                num_segments,
                status_callback=status_callback,
                config=latent_knn_config,
                cancel_token=cancel_token,
                return_posteriors=return_scores,
            )
            if return_scores:
                refined_lowres, lowres_scores = refined_output
            else:
                refined_lowres = refined_output
    elif lowres_labels is not None:
        refined_lowres = lowres_labels.astype(np.uint8, copy=False)
    else:
        refined_lowres = kmeans_outputs[0] if isinstance(kmeans_outputs, tuple) else kmeans_outputs
        if return_scores:
            lowres_scores = _label_to_one_hot(refined_lowres, num_segments)

    coverage_map = torch.tensor(refined_lowres).unsqueeze(0).unsqueeze(0)
    coverage_map = torch.nn.Upsample(
        size=(array.shape[1] + height_pad, array.shape[2] + width_pad), mode="nearest"
    )(coverage_map.byte())

    coverage_map = coverage_map[0, 0, : array.shape[1], : array.shape[2]]

    _emit_status(status_callback, "CNN segmentation map reconstructed.")
    _maybe_raise_cancel(cancel_token)
    labels_full = coverage_map.cpu().numpy()
    labels_full, rotation_plan = _auto_orient_tile_grid(labels_full, tile_size)
    if not return_scores:
        return labels_full

    if lowres_scores is None:
        lowres_scores = _label_to_one_hot(refined_lowres, num_segments)
    score_tensor = torch.from_numpy(lowres_scores).unsqueeze(0)
    score_tensor = F.interpolate(
        score_tensor,
        size=(array.shape[1] + height_pad, array.shape[2] + width_pad),
        mode="bilinear",
        align_corners=False,
    )
    score_tensor = score_tensor[0, :, : array.shape[1], : array.shape[2]]
    scores_np = score_tensor.cpu().numpy()
    if rotation_plan is not None:
        scores_np = _apply_rotation_plan_to_volume(scores_np, tile_size, rotation_plan)
    return labels_full, scores_np


def _compute_latent_grid(
    cnn_model,
    array,
    tile_size,
    device_obj,
    status_callback=None,
    memory_budget=None,
    prefetch_depth=None,
    profile_tier: Optional[str] = None,
    cancel_token=None,
):
    compute_dtype = _runtime_float_dtype(device_obj)
    effective_budget = memory_budget or DEFAULT_MEMORY_BUDGET
    settings = get_adaptive_settings(effective_budget, tier=profile_tier)
    prefetch_depth = prefetch_depth or settings.prefetch_depth
    memory_budget = effective_budget

    tiles, (height_pad, width_pad), grid_shape = tile_raster(array, tile_size)
    _emit_status(
        status_callback,
        f"Raster tiled into {tiles.shape[0]} patches of size {tile_size}x{tile_size}.",
    )

    total_tiles = tiles.shape[0]
    batch_size = _recommended_batch_size(
        tiles.shape[1], tiles.shape[2], tiles.shape[3], memory_budget, settings=settings
    )
    batch_size = max(1, min(batch_size, total_tiles))
    _emit_status(
        status_callback,
        f"Adaptive batch size: {batch_size} tile(s) with prefetch depth {prefetch_depth}.",
    )
    coverage_map = []
    last_report = -1

    def _autocast_ctx():
        if compute_dtype != torch.float16:
            return nullcontext()
        if device_obj.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.amp.autocast(dtype=compute_dtype)
        if device_obj.type == "mps":
            try:
                return torch.autocast(device_type="mps", dtype=compute_dtype)
            except Exception:
                return nullcontext()
        return nullcontext()

    for tensor_batch, start, end, total in _prefetch_batches(
        tiles,
        batch_size,
        device_obj,
        depth=prefetch_depth,
        cancel_token=cancel_token,
        dtype=compute_dtype,
    ):
        _maybe_raise_cancel(cancel_token)
        with torch.no_grad():
            with _autocast_ctx():
                result = cnn_model.forward(tensor_batch)
        vectors = result[1].detach().cpu().numpy()
        coverage_map.append(vectors)
        percent = int((end / total) * 100)
        if percent // 5 > last_report:
            last_report = percent // 5
            _emit_status(status_callback, f"CNN inference {min(percent, 100)}% complete.")
    coverage_map = np.concatenate(coverage_map, axis=0)

    channels = coverage_map.shape[1]
    feat_h = coverage_map.shape[2]
    feat_w = coverage_map.shape[3]
    rows, cols = grid_shape
    coverage_map = coverage_map.transpose(1, 0, 2, 3)
    coverage_map = coverage_map.reshape(
        channels,
        rows,
        cols,
        feat_h,
        feat_w,
    )
    coverage_map = coverage_map.transpose(0, 1, 3, 2, 4)
    coverage_map = coverage_map.reshape(
        channels,
        rows * feat_h,
        cols * feat_w,
    )
    return coverage_map, height_pad, width_pad, grid_shape


def _assign_latent_labels(
    latent_grid: np.ndarray,
    centers: np.ndarray,
    num_segments: int,
    device_obj,
    compute_dtype,
    status_callback=None,
    latent_knn_config: Optional[dict] = None,
    cancel_token=None,
    return_scores: bool = False,
):
    channels, height, width = latent_grid.shape
    descriptors = latent_grid.transpose(1, 2, 0).reshape(-1, channels)
    _emit_status(status_callback, "Assigning CNN latent grid to global centers...")
    labels_flat = _assign_blocks_chunked(descriptors, centers, device_obj, compute_dtype)
    labels_lowres = labels_flat.reshape(height, width).astype(np.uint8, copy=False)
    lowres_scores = None
    if latent_knn_config is not None:
        refined_output = _latent_knn_soft_refine(
            latent_grid,
            labels_lowres,
            centers,
            num_segments,
            status_callback=status_callback,
            config=latent_knn_config,
            cancel_token=cancel_token,
            return_posteriors=return_scores,
        )
        if return_scores:
            labels_lowres, lowres_scores = refined_output
        else:
            labels_lowres = refined_output
    if return_scores and lowres_scores is None:
        lowres_scores = _label_to_one_hot(labels_lowres, num_segments)
    return labels_lowres, lowres_scores


def fit_global_cnn_centers(
    cnn_model,
    array,
    num_segments,
    tile_size,
    device_obj,
    chunk_plan,
    status_callback=None,
    memory_budget=None,
    prefetch_depth=None,
    profile_tier: Optional[str] = None,
    cancel_token=None,
    sample_scale: float = 1.0,
):
    _maybe_raise_cancel(cancel_token)
    compute_dtype = _runtime_float_dtype(device_obj)
    sample_scale = float(np.clip(sample_scale, 0.3, 2.0))
    base_cap = 20000
    max_samples = max(200, int(round(base_cap * sample_scale)))
    rng = np.random.default_rng(0)

    height, width = array.shape[1], array.shape[2]
    should_chunk = bool(chunk_plan and chunk_plan.should_chunk(height, width))
    sampled_chunks = []
    remaining = max_samples

    if not should_chunk:
        latent_grid, _, _, _ = _compute_latent_grid(
            cnn_model,
            array,
            tile_size,
            device_obj,
            status_callback=status_callback,
            memory_budget=memory_budget,
            prefetch_depth=prefetch_depth,
            profile_tier=profile_tier,
            cancel_token=cancel_token,
        )
        blocks_h, blocks_w = latent_grid.shape[1], latent_grid.shape[2]
        descriptors = latent_grid.transpose(1, 2, 0).reshape(-1, latent_grid.shape[0])
        sample_idx = _sample_block_indices((blocks_h, blocks_w), max_samples, rng)
        sampled_chunks.append(descriptors[sample_idx])
    else:
        stride = chunk_plan.stride
        y_starts = _compute_chunk_starts(height, chunk_plan.chunk_size, stride)
        x_starts = _compute_chunk_starts(width, chunk_plan.chunk_size, stride)
        total_chunks = max(1, len(y_starts) * len(x_starts))
        per_chunk = max(1, int(math.ceil(max_samples / total_chunks)))
        _emit_status(
            status_callback,
            f"Sampling global CNN centers across {total_chunks} chunk(s).",
        )
        for y in y_starts:
            for x in x_starts:
                if remaining <= 0:
                    break
                _maybe_raise_cancel(cancel_token)
                y_end = min(y + chunk_plan.chunk_size, height)
                x_end = min(x + chunk_plan.chunk_size, width)
                chunk = array[:, y:y_end, x:x_end]
                latent_grid, _, _, _ = _compute_latent_grid(
                    cnn_model,
                    chunk,
                    tile_size,
                    device_obj,
                    status_callback=None,
                    memory_budget=memory_budget,
                    prefetch_depth=prefetch_depth,
                    profile_tier=profile_tier,
                    cancel_token=cancel_token,
                )
                blocks_h, blocks_w = latent_grid.shape[1], latent_grid.shape[2]
                descriptors = latent_grid.transpose(1, 2, 0).reshape(-1, latent_grid.shape[0])
                sample_cap = min(remaining, per_chunk, descriptors.shape[0])
                sample_idx = _sample_block_indices((blocks_h, blocks_w), sample_cap, rng)
                sampled_chunks.append(descriptors[sample_idx])
                remaining -= sample_cap
            if remaining <= 0:
                break

    if not sampled_chunks:
        raise RuntimeError("Unable to sample latent descriptors for global CNN centers.")
    sampled = np.concatenate(sampled_chunks, axis=0)[:max_samples]
    _emit_status(
        status_callback,
        f"Fitting global CNN centers on {sampled.shape[0]} descriptors (feature_dim={sampled.shape[1]}).",
    )
    centers = _torch_kmeans(
        sampled,
        num_clusters=num_segments,
        device=device_obj,
        compute_dtype=compute_dtype,
        seed=0,
    )
    mapping = _build_label_mapping(centers)
    ordered_centers = _reorder_centers(centers, mapping)
    return ordered_centers


def predict_cnn_with_centers(
    cnn_model,
    array,
    num_segments,
    centers,
    tile_size=256,
    device="cpu",
    status_callback=None,
    memory_budget=None,
    prefetch_depth=None,
    profile_tier: Optional[str] = None,
    latent_knn_config: Optional[dict] = None,
    cancel_token=None,
    return_scores=False,
):
    return predict_cnn(
        cnn_model,
        array,
        num_segments,
        tile_size=tile_size,
        device=device,
        status_callback=status_callback,
        memory_budget=memory_budget,
        prefetch_depth=prefetch_depth,
        profile_tier=profile_tier,
        latent_knn_config=latent_knn_config,
        cancel_token=cancel_token,
        return_scores=return_scores,
        centers=centers,
        require_centers=True,
    )


__all__ = [
    "predict_cnn",
    "predict_cnn_with_centers",
    "fit_global_cnn_centers",
    "tile_raster",
    "_recommended_batch_size",
    "_prefetch_batches",
    "_prefetch_batches_threaded",
    "_prefetch_batches_cuda",
    "_batch_to_tensor",
    "_auto_orient_tile_grid",
    "_apply_rotation_plan_to_volume",
]
