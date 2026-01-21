# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import numpy as np
import torch

import funcs


def _rand(seed, shape):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=shape, dtype=np.uint8)


def test_kmeans_cpu_uses_torch_backend(monkeypatch):
    calls = {"torch": 0}

    def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
        calls["torch"] += 1
        feature_dim = data_np.shape[1]
        return np.stack(
            [np.linspace(0.0, 1.0 + i, feature_dim, dtype=np.float32) for i in range(num_clusters)],
            axis=0,
        )

    def fail_sklearn(*_args, **_kwargs):
        raise AssertionError("sklearn backend must not run")

    monkeypatch.setattr(funcs, "_torch_kmeans", fake_torch_kmeans)
    monkeypatch.setattr(funcs, "_sklearn_kmeans_fit", fail_sklearn)

    array = _rand(0, (3, 32, 32))
    result = funcs.predict_kmeans(array, num_segments=3, resolution=16, device_hint=torch.device("cpu"))

    assert result.shape == array.shape[1:]
    assert result.max() < 3
    assert calls["torch"] == 1


def test_kmeans_respects_torch_distance_path(monkeypatch):
    torch_calls = {"torch": 0, "assign": 0}

    def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
        torch_calls["torch"] += 1
        feature_dim = data_np.shape[1]
        return np.stack(
            [np.linspace(0.0, 1.0 + i, feature_dim, dtype=np.float32) for i in range(num_clusters)],
            axis=0,
        )

    def spy_assign(descriptors, centers, device, compute_dtype, chunk_rows=funcs._DISTANCE_CHUNK_ROWS):
        torch_calls["assign"] += 1
        return np.zeros(descriptors.shape[0], dtype=np.int64)

    def fail_sklearn(*_args, **_kwargs):
        raise AssertionError("sklearn backend must not run")

    monkeypatch.setattr(funcs, "_torch_kmeans", fake_torch_kmeans)
    monkeypatch.setattr(funcs, "_assign_blocks_chunked", spy_assign)
    monkeypatch.setattr(funcs, "_sklearn_kmeans_fit", fail_sklearn)

    array = _rand(1, (3, 24, 24))
    result = funcs.predict_kmeans(array, num_segments=2, resolution=12, device_hint=torch.device("cpu"))

    assert result.shape == array.shape[1:]
    assert result.max() < 2
    assert torch_calls["torch"] == 1
    assert torch_calls["assign"] == 1
