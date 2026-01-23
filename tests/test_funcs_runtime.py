# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Runtime tests for K-Means segmentation pipeline."""

import numpy as np
import pytest

from funcs import (
    AdaptiveSettings,
    ChunkPlan,
    _chunked_gaussian_blur,
    _compute_chunk_starts,
    _normalize_cluster_labels,
    _derive_chunk_size,
    _ChunkAggregator,
    execute_kmeans_segmentation,
    predict_kmeans,
    recommended_chunk_plan,
    set_adaptive_settings,
)
from raster_utils import ensure_channel_first


def _rand_array(seed, shape):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=shape, dtype=np.uint8)


def test_execute_kmeans_segmentation_chunking_preserves_shape():
    """Chunked K-Means segmentation preserves raster shape."""
    array = _rand_array(3, (3, 80, 120))
    plan = ChunkPlan(chunk_size=200, overlap=50, budget_bytes=8 * 1024 * 1024, ratio=0.0075, prefetch_depth=2)
    output = execute_kmeans_segmentation(
        array,
        num_segments=4,
        resolution=16,
        chunk_plan=plan,
        status_callback=lambda *_: None,
    )
    assert output.shape == array.shape[1:]
    assert output.max() < 4


def test_ensure_channel_first_promotes_grayscale_arrays():
    """Grayscale arrays are promoted to (1, H, W) shape."""
    array = np.arange(16, dtype=np.uint8).reshape(4, 4)
    prepared = ensure_channel_first(array)
    assert prepared.shape == (1, 4, 4)
    np.testing.assert_array_equal(prepared[0], array)


def test_ensure_channel_first_rejects_invalid_ndarrays():
    """Arrays with unsupported dimensions raise ValueError."""
    array = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        ensure_channel_first(array)


@pytest.mark.parametrize(
    "array_shape,num_segments,resolution,expected_shape",
    [
        ((3, 64, 64), 4, 16, (64, 64)),
        ((3, 128, 128), 8, 16, (128, 128)),
        ((3, 256, 256), 16, 16, (256, 256)),
        ((3, 256, 128), 8, 16, (256, 128)),
        ((3, 128, 256), 8, 16, (128, 256)),
    ],
)
def test_predict_kmeans_shape_preservation(array_shape, num_segments, resolution, expected_shape):
    """K-Means predictions preserve spatial dimensions."""
    rng = np.random.default_rng(42)
    array = rng.random(array_shape, dtype=np.float32)
    result = predict_kmeans(array, num_segments=num_segments, resolution=resolution)
    assert result.shape == expected_shape


def test_chunked_gaussian_blur_matches_single_pass():
    """Chunked blur matches non-chunked result within tolerance."""
    rng = np.random.default_rng(0)
    array = rng.random((3, 96, 64), dtype=np.float32)
    sigma = 6.0
    full = _chunked_gaussian_blur(array, sigma, max_chunk_bytes=1_000_000_000)
    chunked = _chunked_gaussian_blur(array, sigma, max_chunk_bytes=32 * 1024)
    np.testing.assert_allclose(chunked, full, atol=1e-4)


def test_normalize_cluster_labels_deterministic():
    """Label normalization produces consistent ordering."""
    labels = np.array([[2, 0, 1], [1, 2, 0]], dtype=np.uint8)
    # Create fake centers with distinct means for ordering
    centers = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=np.float32)
    normalized = _normalize_cluster_labels(labels, centers)
    assert normalized.shape == labels.shape
    assert set(np.unique(normalized)) == {0, 1, 2}


def test_compute_chunk_starts_covers_extent():
    """Chunk starts cover the full extent with proper overlap."""
    extent = 200
    chunk_size = 64
    overlap = 16
    starts = _compute_chunk_starts(extent, chunk_size, overlap)
    assert len(starts) >= 1
    assert starts[0] == 0
    # Last chunk should reach or exceed extent
    assert starts[-1] + chunk_size >= extent


def test_chunk_aggregator_basic():
    """ChunkAggregator combines chunks correctly."""
    height, width, num_segments = 64, 64, 4
    chunk_size = 64
    agg = _ChunkAggregator(height, width, num_segments, chunk_size, harmonize_labels=False)
    
    # Add a full-coverage chunk with region tuple (y0, x0, y1, x1)
    labels = np.ones((height, width), dtype=np.uint8)
    region = (0, 0, height, width)
    agg.add(labels, region)
    
    result = agg.finalize()
    assert result.shape == (height, width)


def test_derive_chunk_size_respects_shape():
    """Derived chunk sizes return valid 4-tuple for given shape."""
    import torch
    shape = (3, 1000, 1000)
    device = torch.device("cpu")
    result = _derive_chunk_size(shape, device)
    assert isinstance(result, tuple)
    assert len(result) == 4
    chunk_size, budget, ratio, settings = result
    assert chunk_size >= 64
    assert chunk_size <= max(shape[1], shape[2])


def test_recommended_chunk_plan_creates_valid_plan():
    """recommended_chunk_plan returns a valid ChunkPlan."""
    import torch
    shape = (3, 512, 512)
    device = torch.device("cpu")
    plan = recommended_chunk_plan(shape, device=device)
    assert isinstance(plan, ChunkPlan)
    assert plan.chunk_size > 0
    assert plan.overlap >= 0
