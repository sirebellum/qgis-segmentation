# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Distance and nearest-neighbor utilities."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .common import _coerce_torch_device, _warn_distance_fallback

_DISTANCE_CHUNK_ROWS = 65536


def _prepare_reference_tensor(reference, device, compute_dtype):
    device_obj = _coerce_torch_device(device) or torch.device("cpu")
    try:
        ref = torch.as_tensor(reference, device=device_obj, dtype=compute_dtype)
    except RuntimeError:
        if device_obj.type != "cpu":
            device_obj = torch.device("cpu")
            compute_dtype = torch.float32
            _warn_distance_fallback("device allocation failed; using CPU")
            ref = torch.as_tensor(reference, device=device_obj, dtype=compute_dtype)
        else:
            raise
    ref_norm = ref.float().pow(2).sum(dim=1)
    return ref, ref_norm, device_obj, compute_dtype


def _pairwise_dist2(chunk, reference, reference_norm):
    chunk_norm = chunk.float().pow(2).sum(dim=1, keepdim=True)
    dist2 = chunk_norm + reference_norm.unsqueeze(0)
    dist2 = dist2 - 2.0 * torch.matmul(chunk, reference.transpose(0, 1)).float()
    dist2 = dist2.clamp_min_((0.0))
    if torch.isnan(dist2).any() or torch.isinf(dist2).any():
        raise FloatingPointError("NaN or Inf detected in distance computation.")
    return dist2


def _argmin_distances_chunked(
    data_np: np.ndarray,
    reference: torch.Tensor,
    device: Optional[torch.device],
    compute_dtype: torch.dtype,
    chunk_rows: int = _DISTANCE_CHUNK_ROWS,
):
    ref, ref_norm, device_obj, dtype = _prepare_reference_tensor(reference, device, compute_dtype)
    total = data_np.shape[0]
    labels = np.empty(total, dtype=np.int64)
    step = max(int(chunk_rows), 1)
    try:
        for start in range(0, total, step):
            end = min(start + step, total)
            chunk = torch.as_tensor(data_np[start:end], device=device_obj, dtype=dtype)
            dist2 = _pairwise_dist2(chunk, ref, ref_norm)
            labels[start:end] = torch.argmin(dist2, dim=1).cpu().numpy().astype(np.int64)
    except (RuntimeError, FloatingPointError):
        if dtype == torch.float16:
            _warn_distance_fallback("fp16 distance instability")
            return _argmin_distances_chunked(data_np, reference, device_obj, torch.float32, chunk_rows=step)
        raise
    return labels


def _topk_distances_chunked(
    data_np: np.ndarray,
    reference: torch.Tensor,
    k: int,
    device: Optional[torch.device],
    compute_dtype: torch.dtype,
    chunk_rows: int = _DISTANCE_CHUNK_ROWS,
):
    ref, ref_norm, device_obj, dtype = _prepare_reference_tensor(reference, device, compute_dtype)
    k = max(1, min(int(k), ref.shape[0]))
    total = data_np.shape[0]
    indices = np.empty((total, k), dtype=np.int64)
    step = max(int(chunk_rows), 1)
    max_elements = 4_000_000
    try:
        for start in range(0, total, step):
            end = min(start + step, total)
            chunk = torch.as_tensor(data_np[start:end], device=device_obj, dtype=dtype)
            rows = chunk.shape[0]
            cur_k = min(k, ref.shape[0])
            best_dist = torch.full((rows, cur_k), float("inf"), device=device_obj, dtype=torch.float32)
            best_idx = torch.full((rows, cur_k), -1, device=device_obj, dtype=torch.int64)

            denom = max(rows, 1)
            block_rows = max(1, int(max_elements // denom))
            block_rows = min(block_rows, ref.shape[0])

            ref_start = 0
            while ref_start < ref.shape[0]:
                ref_end = min(ref_start + block_rows, ref.shape[0])
                ref_block = ref[ref_start:ref_end]
                ref_norm_block = ref_norm[ref_start:ref_end]
                dist2 = _pairwise_dist2(chunk, ref_block, ref_norm_block).to(torch.float32)
                offset = torch.arange(ref_start, ref_end, device=device_obj, dtype=torch.int64)
                offset = offset.view(1, -1).expand(rows, -1)

                merged_dist = torch.cat([best_dist, dist2], dim=1)
                merged_idx = torch.cat([best_idx, offset], dim=1)
                best_dist, top_idx = torch.topk(merged_dist, cur_k, largest=False)
                best_idx = torch.gather(merged_idx, 1, top_idx)
                ref_start = ref_end

            indices[start:end] = best_idx.cpu().numpy().astype(np.int64)
    except (RuntimeError, FloatingPointError):
        if dtype == torch.float16:
            _warn_distance_fallback("fp16 distance instability")
            return _topk_distances_chunked(data_np, reference, k, device_obj, torch.float32, chunk_rows=step)
        raise
    return indices


def _assign_labels_tensor(data_tensor: torch.Tensor, centers_tensor: torch.Tensor, compute_dtype: torch.dtype):
    if data_tensor.numel() == 0 or centers_tensor.numel() == 0:
        return torch.zeros(data_tensor.shape[0], device=data_tensor.device, dtype=torch.int64)
    centers = centers_tensor.to(dtype=compute_dtype)
    centers_norm = centers.float().pow(2).sum(dim=1)
    step = max(int(_DISTANCE_CHUNK_ROWS), 1)
    labels = []
    try:
        for start in range(0, data_tensor.shape[0], step):
            end = min(start + step, data_tensor.shape[0])
            chunk = data_tensor[start:end].to(dtype=compute_dtype)
            dist2 = _pairwise_dist2(chunk, centers, centers_norm)
            labels.append(torch.argmin(dist2, dim=1))
    except (RuntimeError, FloatingPointError):
        if compute_dtype == torch.float16:
            _warn_distance_fallback("fp16 distance instability")
            return _assign_labels_tensor(data_tensor, centers_tensor, torch.float32)
        raise
    return torch.cat(labels, dim=0)


def _torch_bruteforce_knn(
    query_np: np.ndarray,
    reference_np: np.ndarray,
    k: int,
    device: Optional[torch.device],
    compute_dtype: torch.dtype,
    chunk_rows: int = _DISTANCE_CHUNK_ROWS,
):
    if query_np.size == 0 or reference_np.size == 0 or k <= 0:
        return np.empty((query_np.shape[0], 0), dtype=np.int64)
    reference_tensor = torch.as_tensor(reference_np)
    return _topk_distances_chunked(
        query_np,
        reference_tensor,
        k,
        device=device,
        compute_dtype=compute_dtype,
        chunk_rows=chunk_rows,
    )


__all__ = [
    "_DISTANCE_CHUNK_ROWS",
    "_prepare_reference_tensor",
    "_pairwise_dist2",
    "_argmin_distances_chunked",
    "_topk_distances_chunked",
    "_assign_labels_tensor",
    "_torch_bruteforce_knn",
]
