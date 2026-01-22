# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import numpy as np
import torch

import funcs
from runtime import kmeans as kmeans_runtime
from runtime.adaptive import ChunkPlan


def test_chunked_kmeans_avoids_per_chunk_normalization(monkeypatch):
    def fail_normalize(*_args, **_kwargs):
        raise AssertionError("Per-chunk label normalization must not run")

    def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
        feature_dim = data_np.shape[1]
        return np.stack(
            [np.full(feature_dim, float(idx), dtype=np.float32) for idx in range(num_clusters)],
            axis=0,
        )

    monkeypatch.setattr(kmeans_runtime, "_normalize_cluster_labels", fail_normalize)
    monkeypatch.setattr(kmeans_runtime, "_torch_kmeans", fake_torch_kmeans)

    array = np.zeros((3, 80, 80), dtype=np.float32)
    labels = funcs.execute_kmeans_segmentation(
        array,
        num_segments=3,
        resolution=16,
        chunk_plan=ChunkPlan(
            chunk_size=32,
            overlap=0,
            budget_bytes=64 * 1024 * 1024,
            ratio=0.01,
            prefetch_depth=1,
        ),
        device_hint=torch.device("cpu"),
    )

    assert labels.shape == array.shape[1:]
    assert labels.max() < 3
