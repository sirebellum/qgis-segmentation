# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Latent-space refinement utilities for CNN outputs."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .common import _distance_compute_dtype, _maybe_raise_cancel, _quantization_device, _runtime_float_dtype, _emit_status
from .distance import _torch_bruteforce_knn

LATENT_KNN_DEFAULTS = {
    "enabled": True,
    "neighbors": 12,
    "temperature": 2.0,
    "mix": 0.65,
    "iterations": 2,
    "spatial_weight": 0.08,
    "chunk_size": 65536,
    "index_points": 120000,
    "query_batch": 32768,
    "hierarchy_factor": 1,
    "hierarchy_passes": 1,
    "mixed_precision_smoothing": True,
}


def _resize_latent_map(latent_map, size):
    dtype = _runtime_float_dtype(None)
    tensor = torch.from_numpy(latent_map).unsqueeze(0).to(dtype=dtype)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).cpu().numpy()


def _resize_label_map(label_map, size):
    dtype = _runtime_float_dtype(None)
    tensor = torch.from_numpy(label_map.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0).to(dtype=dtype)
    resized = F.interpolate(tensor, size=size, mode="nearest")
    return resized.squeeze().cpu().numpy().astype(np.int32, copy=False)


def _stratified_sample_indices(labels, max_points):
    total = labels.size
    if total <= max_points:
        return None
    flat = labels.reshape(-1)
    unique, counts = np.unique(flat, return_counts=True)
    proportions = counts / total
    allocations = np.maximum(1, (proportions * max_points).astype(int))
    diff = allocations.sum() - max_points
    while diff > 0:
        for idx in range(len(allocations)):
            if allocations[idx] > 1 and diff > 0:
                allocations[idx] -= 1
                diff -= 1
    samples = []
    rng = np.random.default_rng(0)
    for value, take in zip(unique, allocations):
        candidates = np.where(flat == value)[0]
        take = min(take, candidates.size)
        if take <= 0:
            continue
        samples.append(rng.choice(candidates, size=take, replace=False))
    if not samples:
        return None
    combined = np.concatenate(samples)
    if combined.size > max_points:
        combined = combined[:max_points]
    return np.sort(combined)


def _latent_knn_soft_refine(
    latent_map,
    lowres_labels,
    centers,
    num_segments,
    status_callback=None,
    config=None,
    cancel_token=None,
    return_posteriors=False,
):
    cfg = dict(LATENT_KNN_DEFAULTS)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    if not cfg.get("enabled", False):
        result = lowres_labels.astype(np.uint8, copy=False)
        if return_posteriors:
            return result, None
        return result

    h, w = lowres_labels.shape
    factor = max(int(cfg.get("hierarchy_factor", 1)), 1)
    passes = max(int(cfg.get("hierarchy_passes", 1)), 1)
    seed_labels = lowres_labels
    if factor > 1 and passes > 1:
        for level in range(passes - 1, 0, -1):
            _maybe_raise_cancel(cancel_token)
            scale = factor * level
            if min(h, w) < scale:
                continue
            coarse_h = max(1, h // scale)
            coarse_w = max(1, w // scale)
            coarse_latent = _resize_latent_map(latent_map, (coarse_h, coarse_w))
            coarse_labels = _resize_label_map(seed_labels, (coarse_h, coarse_w))
            coarse_refined = _latent_knn_core(
                coarse_latent,
                coarse_labels,
                centers,
                num_segments,
                cfg,
                status_callback,
                stage_label=f"coarse@{scale}",
                cancel_token=cancel_token,
            )
            seed_labels = _resize_label_map(coarse_refined, (h, w))

    refined = _latent_knn_core(
        latent_map,
        seed_labels,
        centers,
        num_segments,
        cfg,
        status_callback,
        stage_label="fine",
        cancel_token=cancel_token,
        return_posteriors=return_posteriors,
    )
    return refined


def _latent_knn_core(
    latent_map,
    seed_labels,
    centers,
    num_segments,
    cfg,
    status_callback,
    stage_label="latent",
    cancel_token=None,
    return_posteriors=False,
):
    h, w = seed_labels.shape
    channels = latent_map.shape[0]
    _maybe_raise_cancel(cancel_token)

    vectors = latent_map.reshape(channels, h * w).transpose(1, 0).astype(np.float32, copy=False)
    labels_stack = np.clip(seed_labels.reshape(-1), 0, num_segments - 1)
    coords_y, coords_x = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([coords_y.reshape(-1), coords_x.reshape(-1)], axis=1)
    base_size = vectors.shape[0]
    spatial_weight = float(cfg.get("spatial_weight", 0.0))
    augmented = np.concatenate([vectors, coords * spatial_weight], axis=1)

    temperature = max(float(cfg.get("temperature", 1.0)), 1e-6)
    diffs = vectors[:, None, :] - centers[None, :, :]
    logits = -np.sum(diffs * diffs, axis=2) / temperature
    posteriors = _softmax(logits)

    base = np.eye(num_segments, dtype=np.float32)[labels_stack]
    posteriors = 0.5 * posteriors + 0.5 * base

    neighbors = int(cfg.get("neighbors", 8))
    index_cap = int(cfg.get("index_points", augmented.shape[0]))
    sample_idx = _stratified_sample_indices(labels_stack.reshape(-1), index_cap)
    index_data = augmented if sample_idx is None else augmented[sample_idx]
    knn_device = _quantization_device(None) or torch.device("cpu")
    compute_dtype = _distance_compute_dtype(knn_device)
    query_batch = max(int(cfg.get("query_batch", 32768)), neighbors)
    neighbor_idx = _torch_bruteforce_knn(
        augmented,
        index_data,
        neighbors,
        device=knn_device,
        compute_dtype=compute_dtype,
        chunk_rows=query_batch,
    )
    if sample_idx is not None:
        neighbor_idx = sample_idx[neighbor_idx]

    mix = float(cfg.get("mix", 0.5))
    iterations = max(int(cfg.get("iterations", 1)), 1)
    chunk_size = max(int(cfg.get("chunk_size", 32768)), 1024)

    for iteration in range(iterations):
        neighbor_probs = np.empty_like(posteriors)
        for start in range(0, neighbor_idx.shape[0], chunk_size):
            end = min(start + chunk_size, neighbor_idx.shape[0])
            chunk = neighbor_idx[start:end]
            neighbor_probs[start:end] = posteriors[chunk].mean(axis=1)
        posteriors = (1.0 - mix) * posteriors + mix * neighbor_probs
        percent = int(((iteration + 1) / max(iterations, 1)) * 100)
        _emit_status(
            status_callback,
            f"Latent KNN refinement ({stage_label}) {iteration + 1}/{iterations} ({percent}% complete).",
        )

    refined_base = np.argmax(posteriors[:base_size], axis=1)
    refined_map = refined_base.reshape(h, w).astype(np.uint8)
    if return_posteriors:
        base_scores = posteriors[:base_size]
        scores = base_scores.reshape(h, w, num_segments).transpose(2, 0, 1).astype(np.float32, copy=False)
        return refined_map, scores
    return refined_map


def _softmax(x):
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-8)


__all__ = [
    "LATENT_KNN_DEFAULTS",
    "_latent_knn_soft_refine",
    "_resize_latent_map",
    "_resize_label_map",
    "_stratified_sample_indices",
]
