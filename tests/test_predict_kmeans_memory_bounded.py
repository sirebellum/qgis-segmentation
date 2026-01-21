# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import numpy as np

import funcs


def test_predict_kmeans_uses_pooled_descriptors(monkeypatch):
    shapes = {}

    real_pool = funcs._smooth_and_pool_descriptors

    def wrapped_pool(array, resolution, device, compute_dtype, status_callback=None, cancel_token=None):
        desc, block_shape, height_pad, width_pad = real_pool(
            array,
            resolution,
            device,
            compute_dtype,
            status_callback=status_callback,
            cancel_token=cancel_token,
        )
        shapes["desc_shape"] = desc.shape
        shapes["block_shape"] = block_shape
        shapes["height_pad"] = height_pad
        shapes["width_pad"] = width_pad
        return desc, block_shape, height_pad, width_pad

    def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
        # Return simple linearly separated centers to avoid random behavior during the test.
        feature_dim = data_np.shape[1]
        return np.stack(
            [np.linspace(0.0, float(i + 1), feature_dim, dtype=np.float32) for i in range(num_clusters)],
            axis=0,
        )

    def spy_assign(descriptors, centers, device, compute_dtype, chunk_rows=funcs._DISTANCE_CHUNK_ROWS):
        shapes["assign_shape"] = descriptors.shape
        return np.zeros(descriptors.shape[0], dtype=np.int64)

    monkeypatch.setattr(funcs, "_smooth_and_pool_descriptors", wrapped_pool)
    monkeypatch.setattr(funcs, "_torch_kmeans", fake_torch_kmeans)
    monkeypatch.setattr(funcs, "_assign_blocks_chunked", spy_assign)

    rng = np.random.default_rng(123)
    array = rng.random((3, 48, 32), dtype=np.float32)

    result = funcs.predict_kmeans(array, num_segments=4, resolution=16)

    # Descriptor dimension should match channels, not channels*resolution*resolution.
    assert shapes["desc_shape"][1] == array.shape[0]
    assert shapes["assign_shape"][1] == array.shape[0]
    # Number of descriptors should equal pooled grid size.
    pooled_h, pooled_w = shapes["block_shape"]
    assert shapes["desc_shape"][0] == pooled_h * pooled_w
    assert result.shape == array.shape[1:]
    assert result.max() < 4
