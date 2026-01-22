# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import numpy as np
import torch

import funcs
from runtime import kmeans as kmeans_runtime
from runtime.adaptive import ChunkPlan


def _chunk_plan(chunk_size=32):
    return ChunkPlan(
        chunk_size=chunk_size,
        overlap=0,
        budget_bytes=64 * 1024 * 1024,
        ratio=0.01,
        prefetch_depth=1,
    )


def test_global_kmeans_fit_called_once(monkeypatch):
    calls = {"torch": 0}

    def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
        calls["torch"] += 1
        feature_dim = data_np.shape[1]
        return np.stack(
            [np.full(feature_dim, float(idx), dtype=np.float32) for idx in range(num_clusters)],
            axis=0,
        )

    monkeypatch.setattr(kmeans_runtime, "_torch_kmeans", fake_torch_kmeans)

    array = np.zeros((3, 96, 96), dtype=np.float32)
    labels = funcs.execute_kmeans_segmentation(
        array,
        num_segments=3,
        resolution=16,
        chunk_plan=_chunk_plan(),
        device_hint=torch.device("cpu"),
    )

    assert labels.shape == array.shape[1:]
    assert labels.max() < 3
    assert calls["torch"] == 1


def test_global_centers_keep_labels_consistent_across_chunks(monkeypatch):
    def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(kmeans_runtime, "_torch_kmeans", fake_torch_kmeans)

    array = np.zeros((3, 64, 64), dtype=np.float32)
    array[0, :, :] = 1.0

    labels = funcs.execute_kmeans_segmentation(
        array,
        num_segments=2,
        resolution=16,
        chunk_plan=_chunk_plan(chunk_size=32),
        device_hint=torch.device("cpu"),
    )

    left = labels[:, : labels.shape[1] // 2]
    right = labels[:, labels.shape[1] // 2 :]

    assert labels.max() < 2
    assert np.array_equal(np.unique(left), np.unique(right))
