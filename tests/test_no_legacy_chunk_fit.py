# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import numpy as np
import torch

from runtime import chunking as chunking_runtime
from runtime import cnn as cnn_runtime
from runtime import kmeans as kmeans_runtime
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


def test_chunked_cnn_skips_legacy_fit_and_harmonize(monkeypatch):
    def fail_predict_kmeans(*_args, **_kwargs):
        raise AssertionError("Legacy per-chunk K-Means fit must not run")

    def fail_harmonize(*_args, **_kwargs):
        raise AssertionError("Per-chunk label harmonization must not run")

    monkeypatch.setattr(kmeans_runtime, "predict_kmeans", fail_predict_kmeans)
    monkeypatch.setattr(chunking_runtime._ChunkAggregator, "_harmonize_labels", fail_harmonize)

    array = np.zeros((3, 96, 96), dtype=np.float32)
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
    assert labels.max() < 2
