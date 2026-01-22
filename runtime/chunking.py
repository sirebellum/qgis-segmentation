# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Chunked processing and blending helpers."""
from __future__ import annotations

import itertools
from typing import Callable, Optional, Tuple

import numpy as np

from .common import _emit_status, _maybe_raise_cancel
from .smoothing import _build_weight_mask, _gaussian_blur_channels


def _compute_chunk_starts(length, chunk_size, stride):
    if length <= chunk_size:
        return [0]
    starts = list(range(0, max(1, length - chunk_size), stride))
    last_start = length - chunk_size
    if last_start > 0 and (not starts or starts[-1] != last_start):
        starts.append(last_start)
    return sorted(set(starts))


def _normalize_inference_output(result):
    scores = None
    labels = result
    if isinstance(result, dict):
        labels = result.get("labels")
        scores = result.get("scores")
    elif isinstance(result, (list, tuple)) and len(result) == 2:
        labels, scores = result
    if labels is None:
        raise ValueError("Inference result must include labels.")
    return labels, scores


def _label_to_one_hot(label_map, num_segments):
    labels = np.clip(label_map.astype(np.int64, copy=False), 0, num_segments - 1)
    one_hot = np.eye(num_segments, dtype=np.float32)[labels]
    return one_hot.transpose(2, 0, 1)


def _process_in_chunks(
    array,
    plan,
    num_segments,
    infer_fn: Callable,
    status_callback,
    smoothing_scale=1.0,
    cancel_token=None,
    harmonize_labels: bool = True,
):
    height, width = array.shape[1], array.shape[2]
    if not plan.should_chunk(height, width):
        return infer_fn(array)

    stride = plan.stride
    y_starts = _compute_chunk_starts(height, plan.chunk_size, stride)
    x_starts = _compute_chunk_starts(width, plan.chunk_size, stride)
    total = len(y_starts) * len(x_starts)
    aggregator = _ChunkAggregator(
        height,
        width,
        num_segments,
        plan.chunk_size,
        status_callback=status_callback,
        smoothing_scale=smoothing_scale,
        harmonize_labels=harmonize_labels,
    )

    for idx, (y, x) in enumerate(itertools.product(y_starts, x_starts), start=1):
        _maybe_raise_cancel(cancel_token)
        y_end = min(y + plan.chunk_size, height)
        x_end = min(x + plan.chunk_size, width)
        chunk = array[:, y:y_end, x:x_end]
        if status_callback:
            status_callback(f"Chunk {idx}/{total}: rows {y}-{y_end}, cols {x}-{x_end}")
        inference_result = infer_fn(chunk)
        labels, scores = _normalize_inference_output(inference_result)
        aggregator.add(labels, (y, x, y_end, x_end), chunk_data=chunk, scores=scores)

    if status_callback:
        megapixels = (height * width) / 1_000_000
        status_callback(
            f"Smoothing out {total} chunks (~{megapixels:.2f} MP of coverage)..."
        )
    _maybe_raise_cancel(cancel_token)
    return aggregator.finalize()


class _ChunkAggregator:
    def __init__(
        self,
        height,
        width,
        num_segments,
        chunk_size,
        status_callback=None,
        smoothing_scale=1.0,
        harmonize_labels: bool = True,
    ):
        self.height = height
        self.width = width
        self.num_segments = num_segments
        self.scores = np.zeros((num_segments, height, width), dtype=np.float32)
        self.weight = np.zeros((height, width), dtype=np.float32)
        self.weight_template = _build_weight_mask(chunk_size)
        self.chunk_size = chunk_size
        self._status_callback = status_callback
        self._smoothing_scale = max(0.1, float(smoothing_scale))
        self._palette_threshold = 24.0
        self._feature_dim = None
        self._prototype_vectors = None
        self._prototype_counts = np.zeros(num_segments, dtype=np.int64)
        self._harmonize_labels = bool(harmonize_labels)

    def add(self, labels, region, chunk_data=None, scores=None):
        y0, x0, y1, x1 = region
        h = y1 - y0
        w = x1 - x0
        if self._harmonize_labels:
            harmonized = self._harmonize_labels(labels[:h, :w], chunk_data)
        else:
            harmonized = labels[:h, :w]
        chunk = harmonized.astype(np.int64, copy=False)
        mask = self.weight_template[:h, :w]
        if scores is not None:
            chunk_scores = scores[:, :h, :w].astype(np.float32, copy=False)
            weighted_scores = chunk_scores * mask[np.newaxis, ...]
        else:
            one_hot = np.eye(self.num_segments, dtype=np.float32)[chunk]
            chunk_scores = one_hot.transpose(2, 0, 1)
            weighted_scores = chunk_scores * mask
        self.scores[:, y0:y1, x0:x1] += weighted_scores
        self.weight[y0:y1, x0:x1] += mask

    def finalize(self):
        sigma = max(1.0, min((self.chunk_size / 10.0) * self._smoothing_scale, 32.0))
        _emit_status(self._status_callback, "Smoothing CNN logits with GPU gradients...")
        smoothed_scores = _gaussian_blur_channels(
            self.scores,
            sigma,
            status_callback=lambda msg: _emit_status(self._status_callback, msg),
            stage_label="scores",
        )
        smoothed_weight = _gaussian_blur_channels(
            self.weight[np.newaxis, ...],
            sigma,
            status_callback=lambda msg: _emit_status(self._status_callback, msg),
            stage_label="weights",
        )[0]
        weight = np.maximum(smoothed_weight, 1e-6)
        probs = smoothed_scores / weight
        return np.argmax(probs, axis=0).astype(np.uint8)

    def _harmonize_labels(self, labels, chunk_data):
        if chunk_data is None:
            return labels
        label_vectors = self._extract_label_vectors(chunk_data, labels)
        if not label_vectors:
            return labels
        if self._feature_dim is None:
            sample = next(iter(label_vectors.values()))
            self._feature_dim = sample.shape[0]
            self._prototype_vectors = np.zeros((self.num_segments, self._feature_dim), dtype=np.float32)
        remapped = labels.copy()
        used_targets = set()
        for label_id, vector in label_vectors.items():
            target = self._select_target_index(vector, used_targets)
            if target is None:
                continue
            used_targets.add(target)
            self._update_prototype(target, vector)
            if target != label_id:
                remapped[labels == label_id] = target
        return remapped

    def _extract_label_vectors(self, chunk_data, labels):
        if chunk_data.ndim != 3:
            return {}
        channels = chunk_data.shape[0]
        flat_pixels = chunk_data.reshape(channels, -1).astype(np.float32, copy=False)
        flat_labels = labels.reshape(-1)
        vectors = {}
        for label_id in np.unique(flat_labels):
            mask = flat_labels == label_id
            if not np.any(mask):
                continue
            pixels = flat_pixels[:, mask]
            vectors[int(label_id)] = pixels.mean(axis=1)
        return vectors

    def _select_target_index(self, vector, used_targets):
        if self._prototype_vectors is None:
            return self._next_unused_index(used_targets)
        active_indices = [idx for idx, count in enumerate(self._prototype_counts) if count > 0 and idx not in used_targets]
        candidate = None
        min_distance = None
        for idx in active_indices:
            distance = float(np.linalg.norm(self._prototype_vectors[idx] - vector))
            if min_distance is None or distance < min_distance:
                min_distance = distance
                candidate = idx
        if candidate is not None and min_distance is not None and min_distance <= self._palette_threshold:
            return candidate
        fallback = self._next_unused_index(used_targets)
        if fallback is not None:
            return fallback
        return candidate

    def _update_prototype(self, idx, vector):
        if self._prototype_vectors is None:
            return
        current = self._prototype_counts[idx]
        vector = vector.astype(np.float32, copy=False)
        if current == 0:
            self._prototype_vectors[idx] = vector
        else:
            total = current + 1
            self._prototype_vectors[idx] = (self._prototype_vectors[idx] * current + vector) / total
        self._prototype_counts[idx] = current + 1

    def _next_unused_index(self, used_targets):
        for idx in range(self.num_segments):
            if self._prototype_counts[idx] == 0 and idx not in used_targets:
                return idx
        for idx in range(self.num_segments):
            if idx not in used_targets:
                return idx
        return None


__all__ = [
    "_compute_chunk_starts",
    "_normalize_inference_output",
    "_label_to_one_hot",
    "_process_in_chunks",
    "_ChunkAggregator",
]
