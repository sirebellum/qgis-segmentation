# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

from funcs import (
    _argmin_distances_chunked,
    _distance_compute_dtype,
    _torch_bruteforce_knn,
    _torch_kmeans,
)
import numpy as np
import os
import pytest

torch = pytest.importorskip("torch")


def _sorted_centers(centers):
    order = np.argsort(centers.sum(axis=1))
    return centers[order]


def test_torch_kmeans_deterministic_cpu():
    data_a = np.full((8, 2), -1.0, dtype=np.float32)
    data_b = np.full((8, 2), 3.0, dtype=np.float32)
    data = np.vstack([data_a, data_b])
    centers_1 = _torch_kmeans(data, num_clusters=2, device=torch.device(
        "cpu"), compute_dtype=torch.float32, seed=42)
    centers_2 = _torch_kmeans(data, num_clusters=2, device=torch.device(
        "cpu"), compute_dtype=torch.float32, seed=42)
    np.testing.assert_allclose(_sorted_centers(
        centers_1), _sorted_centers(centers_2), atol=1e-5)


def test_torch_knn_matches_bruteforce_small():
    reference = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    query = np.array([[0.1, 0.1], [1.9, 2.1]], dtype=np.float32)
    expected = np.argsort(
        ((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2), axis=1)[:, :2]
    indices = _torch_bruteforce_knn(
        query,
        reference,
        k=2,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        chunk_rows=2,
    )
    np.testing.assert_array_equal(indices, expected)


def test_distance_chunking_equivalence():
    rng = np.random.default_rng(123)
    data = rng.random((64, 4), dtype=np.float32)
    centers = rng.random((3, 4), dtype=np.float32)
    baseline = _argmin_distances_chunked(data, torch.as_tensor(
        centers), torch.device("cpu"), torch.float32, chunk_rows=128)
    chunked = _argmin_distances_chunked(data, torch.as_tensor(
        centers), torch.device("cpu"), torch.float32, chunk_rows=7)
    np.testing.assert_array_equal(baseline, chunked)
