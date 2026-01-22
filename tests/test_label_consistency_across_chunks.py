# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import numpy as np
import torch

from runtime import cnn as cnn_runtime
from runtime.adaptive import ChunkPlan
from runtime.pipeline import execute_cnn_segmentation


class _FakeModel:
    def __init__(self, channels=2):
        self._channels = channels

    def to(self, _device):
        return self

    def eval(self):
        return self

    def forward(self, batch):
        mean = batch.mean(dim=(2, 3), keepdim=True)
        latent = mean.repeat(1, self._channels, 1, 1)
        return (None, latent)


def test_label_consistency_across_chunks(monkeypatch):
    def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
        feature_dim = data_np.shape[1]
        return np.stack(
            [np.zeros(feature_dim, dtype=np.float32), np.ones(feature_dim, dtype=np.float32)],
            axis=0,
        )

    monkeypatch.setattr(cnn_runtime, "_torch_kmeans", fake_torch_kmeans)

    array = np.zeros((3, 64, 64), dtype=np.float32)
    array[:, :, 32:] = 1.0

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

    left = labels[:, : labels.shape[1] // 2]
    right = labels[:, labels.shape[1] // 2 :]

    assert labels.max() < 2
    assert np.all(left == left[0, 0])
    assert np.all(right == right[0, 0])
