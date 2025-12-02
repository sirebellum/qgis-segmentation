import numpy as np
import pytest
import torch

from funcs import (
    tile_raster,
    predict_cnn,
    execute_cnn_segmentation,
    execute_kmeans_segmentation,
    ChunkPlan,
    AdaptiveSettings,
    set_adaptive_settings,
    _compute_chunk_starts,
    _normalize_cluster_labels,
    recommended_chunk_plan,
    _derive_chunk_size,
    _ChunkAggregator,
)
from perf_tuner import load_or_profile_settings
from raster_utils import ensure_channel_first


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_history = []

    def forward(self, x):
        self.batch_history.append(x.shape[0])
        mean = x.mean(dim=1, keepdim=True)
        maxv, _ = x.max(dim=1, keepdim=True)
        minv, _ = x.min(dim=1, keepdim=True)
        stacked = torch.cat([mean, maxv, minv], dim=1)
        return (None, stacked)


def _rand_array(seed, shape):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=shape, dtype=np.uint8)


def test_tile_raster_reassembles_original_extent():
    array = _rand_array(0, (3, 133, 287))
    tile_size = 64
    tiles, pads, grid = tile_raster(array, tile_size=tile_size)
    rows, cols = grid
    recon = tiles.reshape(rows, cols, array.shape[0], tile_size, tile_size)
    recon = recon.transpose(2, 0, 3, 1, 4).reshape(array.shape[0], rows * tile_size, cols * tile_size)
    recon = recon[:, : array.shape[1], : array.shape[2]]
    np.testing.assert_array_equal(array, recon)
    expected_pads = (
        (tile_size - (array.shape[1] % tile_size)) % tile_size,
        (tile_size - (array.shape[2] % tile_size)) % tile_size,
    )
    assert pads == expected_pads


@pytest.mark.parametrize(
    "shape,tile_size",
    [
        ((3, 384, 640), 128),
        ((3, 257, 513), 96),
    ],
)
def test_predict_cnn_handles_rectangular_tiles(shape, tile_size):
    array = _rand_array(1, shape)
    model = _DummyModel()
    result = predict_cnn(
        model,
        array,
        num_segments=3,
        tile_size=tile_size,
        device="cpu",
        memory_budget=512 * 1024 * 1024,
    )
    assert result.shape == shape[1:]
    assert result.dtype == np.uint8
    assert max(model.batch_history) >= 1


def test_predict_cnn_uses_adaptive_batching():
    shape = (3, 256, 512)
    array = _rand_array(4, shape)
    model = _DummyModel()
    predict_cnn(
        model,
        array,
        num_segments=3,
        tile_size=64,
        device="cpu",
        memory_budget=1024 * 1024 * 1024,
        prefetch_depth=3,
    )
    assert max(model.batch_history) > 1


@pytest.mark.parametrize(
    "shape",
    [
        (3, 640, 925),
        (3, 512, 700),
    ],
)
def test_execute_cnn_segmentation_chunking_preserves_shape(shape):
    array = _rand_array(2, shape)
    plan = ChunkPlan(chunk_size=256, overlap=64, budget_bytes=128 * 1024 * 1024, ratio=0.0075, prefetch_depth=2)
    output = execute_cnn_segmentation(
        _DummyModel(),
        array,
        num_segments=3,
        chunk_plan=plan,
        tile_size=256,
        device=torch.device("cpu"),
    )
    assert output.shape == shape[1:]
    assert output.max() < 3


def test_execute_kmeans_segmentation_chunking_preserves_shape():
    array = _rand_array(3, (3, 781, 1023))
    plan = ChunkPlan(chunk_size=200, overlap=50, budget_bytes=64 * 1024 * 1024, ratio=0.0075, prefetch_depth=2)
    output = execute_kmeans_segmentation(
        array,
        num_segments=4,
        resolution=16,
        chunk_plan=plan,
        status_callback=lambda *_: None,
    )
    assert output.shape == array.shape[1:]
    assert output.max() < 4


def test_perf_tuner_profiles_and_caches(tmp_path):
    device = torch.device("cpu")
    calls = []

    def fake_runner(dev, status):
        calls.append(dev.type)
        return AdaptiveSettings(safety_factor=5, prefetch_depth=3)

    settings, created = load_or_profile_settings(
        tmp_path, device, benchmark_runner=fake_runner
    )
    assert created
    assert settings.prefetch_depth == 3
    assert settings.safety_factor == 5
    cached, created2 = load_or_profile_settings(
        tmp_path, device, benchmark_runner=fake_runner
    )
    assert not created2
    assert cached.prefetch_depth == 3
    assert len(calls) == 1
    set_adaptive_settings(AdaptiveSettings())


def test_ensure_channel_first_promotes_grayscale_arrays():
    array = np.arange(16, dtype=np.uint8).reshape(4, 4)
    prepared = ensure_channel_first(array)
    assert prepared.shape == (1, 4, 4)
    np.testing.assert_array_equal(prepared[0], array)


def test_ensure_channel_first_rejects_invalid_ndarrays():
    array = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        ensure_channel_first(array)


# Tests for _compute_chunk_starts
class TestComputeChunkStarts:
    def test_small_length_returns_single_start(self):
        """When length <= chunk_size, should return [0]."""
        result = _compute_chunk_starts(length=100, chunk_size=100, stride=50)
        assert result == [0]

        result = _compute_chunk_starts(length=50, chunk_size=100, stride=50)
        assert result == [0]

    def test_computes_correct_starts_with_stride(self):
        """Basic case with chunk_size and stride."""
        result = _compute_chunk_starts(length=300, chunk_size=100, stride=80)
        # Expected starts: 0, 80, 160, and last_start = 300-100 = 200
        assert 0 in result
        assert 200 in result  # last_start should be included
        assert result == sorted(set(result))  # Should be sorted and unique

    def test_includes_last_start_when_not_in_range(self):
        """Ensure the last chunk start is included to cover the full length."""
        result = _compute_chunk_starts(length=250, chunk_size=100, stride=100)
        # With stride=100, starts would be [0, 100], but last_start = 250-100 = 150
        assert 150 in result

    def test_handles_exact_multiple(self):
        """When length is exact multiple of stride."""
        result = _compute_chunk_starts(length=200, chunk_size=100, stride=100)
        assert 0 in result
        assert 100 in result  # last_start = 200-100 = 100

    def test_deduplication_and_sorting(self):
        """Results should be deduplicated and sorted."""
        result = _compute_chunk_starts(length=200, chunk_size=100, stride=50)
        assert result == sorted(set(result))


# Tests for _normalize_cluster_labels
class TestNormalizeClusterLabels:
    def test_reorders_by_center_mean(self):
        """Labels should be reordered based on cluster center means."""
        labels = np.array([0, 1, 2, 0, 1, 2])
        # Centers with means: cluster 0 = 30, cluster 1 = 10, cluster 2 = 20
        centers = np.array([[30.0], [10.0], [20.0]])
        result = _normalize_cluster_labels(labels, centers)
        # Order by mean: 1 (10) -> 0, 2 (20) -> 1, 0 (30) -> 2
        expected = np.array([2, 0, 1, 2, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_preserves_shape(self):
        """Output shape should match input shape."""
        labels = np.array([[0, 1], [2, 0]])
        centers = np.array([[1.0], [2.0], [3.0]])
        result = _normalize_cluster_labels(labels, centers)
        assert result.shape == labels.shape

    def test_handles_single_cluster(self):
        """Edge case with single cluster."""
        labels = np.array([0, 0, 0, 0])
        centers = np.array([[5.0]])
        result = _normalize_cluster_labels(labels, centers)
        np.testing.assert_array_equal(result, np.array([0, 0, 0, 0]))

    def test_raises_on_invalid_labels(self):
        """Should raise ValueError for out-of-bounds labels."""
        labels = np.array([0, 1, 3])  # 3 is out of bounds
        centers = np.array([[1.0], [2.0], [3.0]])  # Only 3 centers
        with pytest.raises(ValueError, match="out-of-bounds"):
            _normalize_cluster_labels(labels, centers)

    def test_raises_on_negative_labels(self):
        """Should raise ValueError for negative labels."""
        labels = np.array([0, -1, 1])
        centers = np.array([[1.0], [2.0]])
        with pytest.raises(ValueError, match="out-of-bounds"):
            _normalize_cluster_labels(labels, centers)


# Tests for recommended_chunk_plan
class TestRecommendedChunkPlan:
    def test_returns_chunk_plan(self):
        """Should return a valid ChunkPlan object."""
        array_shape = (3, 1024, 1024)
        device = torch.device("cpu")
        plan = recommended_chunk_plan(array_shape, device)
        assert isinstance(plan, ChunkPlan)
        assert plan.chunk_size >= 128
        assert plan.overlap > 0
        assert plan.overlap < plan.chunk_size

    def test_overlap_is_quarter_of_chunk_size(self):
        """Overlap should be approximately 25% of chunk_size (minimum 32)."""
        array_shape = (3, 1024, 1024)
        device = torch.device("cpu")
        plan = recommended_chunk_plan(array_shape, device)
        expected_overlap = max(32, int(plan.chunk_size * 0.25))
        assert plan.overlap == expected_overlap

    def test_respects_adaptive_settings(self):
        """Should use prefetch_depth from adaptive settings."""
        original_settings = AdaptiveSettings()
        try:
            set_adaptive_settings(AdaptiveSettings(prefetch_depth=5))
            array_shape = (3, 512, 512)
            device = torch.device("cpu")
            plan = recommended_chunk_plan(array_shape, device)
            assert plan.prefetch_depth == 5
        finally:
            set_adaptive_settings(original_settings)


# Tests for _derive_chunk_size
class TestDeriveChunkSize:
    def test_returns_tuple_of_three(self):
        """Should return (chunk_size, budget, ratio)."""
        array_shape = (3, 512, 512)
        device = torch.device("cpu")
        result = _derive_chunk_size(array_shape, device)
        assert len(result) == 3
        chunk_size, budget, ratio = result
        assert isinstance(chunk_size, int)
        assert isinstance(budget, int)
        assert isinstance(ratio, float)

    def test_chunk_size_within_bounds(self):
        """Chunk size should be between 128 and 512."""
        array_shape = (3, 1024, 1024)
        device = torch.device("cpu")
        chunk_size, _, _ = _derive_chunk_size(array_shape, device)
        assert 128 <= chunk_size <= 512

    def test_budget_is_positive(self):
        """Budget should be a positive value."""
        array_shape = (3, 512, 512)
        device = torch.device("cpu")
        _, budget, _ = _derive_chunk_size(array_shape, device)
        assert budget > 0

    def test_different_channel_counts(self):
        """Should handle arrays with different channel counts."""
        device = torch.device("cpu")
        # More channels means more memory per pixel
        result_3ch = _derive_chunk_size((3, 512, 512), device)
        result_6ch = _derive_chunk_size((6, 512, 512), device)
        # Both should return valid results
        assert result_3ch[0] >= 128
        assert result_6ch[0] >= 128


# Tests for _ChunkAggregator
class TestChunkAggregator:
    def test_single_chunk_preserves_center_labels(self):
        """Single chunk should preserve labels in the center (high weight region)."""
        height, width = 64, 64
        num_segments = 3
        chunk_size = 64
        agg = _ChunkAggregator(height, width, num_segments, chunk_size)
        # Use uniform labels to test basic functionality
        labels = np.full((height, width), 1, dtype=np.uint8)
        agg.add(labels, (0, 0, height, width))
        result = agg.finalize()
        # Center region should preserve the label
        center_slice = result[20:44, 20:44]
        np.testing.assert_array_equal(center_slice, np.full_like(center_slice, 1))

    def test_output_shape_matches_dimensions(self):
        """Finalized output should have correct shape."""
        height, width = 100, 150
        num_segments = 5
        chunk_size = 64
        agg = _ChunkAggregator(height, width, num_segments, chunk_size)
        # Add a small chunk
        labels = np.zeros((64, 64), dtype=np.uint8)
        agg.add(labels, (0, 0, 64, 64))
        result = agg.finalize()
        assert result.shape == (height, width)

    def test_overlapping_chunks_are_blended(self):
        """Overlapping chunks should be blended using weights."""
        height, width = 100, 100
        num_segments = 2
        chunk_size = 64
        agg = _ChunkAggregator(height, width, num_segments, chunk_size)
        # Add two overlapping chunks with different labels
        labels1 = np.zeros((64, 64), dtype=np.uint8)  # All label 0
        labels2 = np.ones((64, 64), dtype=np.uint8)   # All label 1
        agg.add(labels1, (0, 0, 64, 64))
        agg.add(labels2, (36, 36, 100, 100))  # Overlaps from (36,36) to (64,64)
        result = agg.finalize()
        # The result should be valid segmentation with values 0 or 1
        assert result.min() >= 0
        assert result.max() < num_segments

    def test_handles_partial_chunks_at_edges(self):
        """Should handle chunks smaller than chunk_size at edges."""
        height, width = 80, 80
        num_segments = 3
        chunk_size = 64
        agg = _ChunkAggregator(height, width, num_segments, chunk_size)
        # Add chunks with partial coverage
        labels1 = np.zeros((64, 64), dtype=np.uint8)
        labels2 = np.ones((64, 16), dtype=np.uint8)  # Partial width
        labels3 = np.full((16, 64), 2, dtype=np.uint8)  # Partial height
        labels4 = np.full((16, 16), 1, dtype=np.uint8)  # Corner
        agg.add(labels1, (0, 0, 64, 64))
        agg.add(labels2, (0, 64, 64, 80))
        agg.add(labels3, (64, 0, 80, 64))
        agg.add(labels4, (64, 64, 80, 80))
        result = agg.finalize()
        assert result.shape == (height, width)
        assert result.dtype == np.uint8

    def test_finalize_returns_argmax_of_weighted_scores(self):
        """The finalize method should return argmax of weighted scores."""
        height, width = 32, 32
        num_segments = 2
        chunk_size = 32
        agg = _ChunkAggregator(height, width, num_segments, chunk_size)
        # Add label 0 twice (more weight)
        labels0 = np.zeros((32, 32), dtype=np.uint8)
        labels1 = np.ones((32, 32), dtype=np.uint8)
        agg.add(labels0, (0, 0, 32, 32))
        agg.add(labels0, (0, 0, 32, 32))
        agg.add(labels1, (0, 0, 32, 32))
        result = agg.finalize()
        # Label 0 should win because it has more weight
        np.testing.assert_array_equal(result, labels0)
