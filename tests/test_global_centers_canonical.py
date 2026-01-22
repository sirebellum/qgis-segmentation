# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import numpy as np
import torch

from runtime import cnn as cnn_runtime
from runtime.adaptive import ChunkPlan
from runtime.pipeline import execute_cnn_segmentation


class _FakeModel:
    def __init__(self, channels=3):
        self._channels = channels

    def to(self, _device):
        return self

    def eval(self):
        return self

    def forward(self, batch):
        mean = batch.mean(dim=(2, 3), keepdim=True)
        latent = mean.repeat(1, self._channels, 1, 1)
        return (None, latent)


def test_global_cnn_centers_fit_once(monkeypatch):
    calls = {"torch": 0}

    def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
        calls["torch"] += 1
        feature_dim = data_np.shape[1]
        return np.stack(
            [np.full(feature_dim, float(idx), dtype=np.float32) for idx in range(num_clusters)],
            axis=0,
        )

    monkeypatch.setattr(cnn_runtime, "_torch_kmeans", fake_torch_kmeans)

    array = np.zeros((3, 96, 96), dtype=np.float32)
    labels = execute_cnn_segmentation(
        _FakeModel(),
        array,
        num_segments=3,
        chunk_plan=ChunkPlan(
            chunk_size=32,
            overlap=0,
            budget_bytes=64 * 1024 * 1024,
            ratio=0.01,
            prefetch_depth=1,
        ),
        tile_size=32,
        device=torch.device("cpu"),
    )

    assert labels.shape == array.shape[1:]
    assert labels.max() < 3
    assert calls["torch"] == 1


def test_centers_identity_across_chunks(monkeypatch):
    center_ids = []
    original_assign = cnn_runtime._assign_blocks_chunked

    def spy_assign(descriptors, centers, device, compute_dtype, chunk_rows=cnn_runtime._DISTANCE_CHUNK_ROWS):
        center_ids.append(id(centers))
        return original_assign(descriptors, centers, device, compute_dtype, chunk_rows=chunk_rows)

    monkeypatch.setattr(cnn_runtime, "_assign_blocks_chunked", spy_assign)

    array = np.zeros((3, 128, 96), dtype=np.float32)
    labels = execute_cnn_segmentation(
        _FakeModel(),
        array,
        num_segments=2,
        chunk_plan=ChunkPlan(
            chunk_size=32,
            overlap=0,
            budget_bytes=64 * 1024 * 1024,
            ratio=0.01,
            prefetch_depth=1,
        ),
        tile_size=32,
        device=torch.device("cpu"),
    )

    assert labels.shape == array.shape[1:]
    assert len(center_ids) > 1
    assert len(set(center_ids)) == 1
