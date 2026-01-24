# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Tests for seam prevention: halo overlap, global alignment, and consistent labels.

These tests verify that chunk boundary seams are eliminated by:
1. Halo overlap: edge pixels have proper smoothing context
2. Global block alignment: block indices are computed from absolute coordinates
3. Consistent label assignment across chunks using global centers
"""

import numpy as np
import pytest
import torch

import funcs
from runtime import kmeans as kmeans_runtime
from runtime.adaptive import ChunkPlan
from runtime.chunking import (
    DEFAULT_HALO_PIXELS,
    _expand_window_for_halo,
    _crop_halo_from_result,
)
from runtime.kmeans import (
    DESCRIPTOR_HALO_PIXELS,
    BLOCK_OVERLAP,
    _block_grid_shape,
    _compute_globally_aligned_descriptors,
)


def _chunk_plan(chunk_size=32, overlap=0):
    return ChunkPlan(
        chunk_size=chunk_size,
        overlap=overlap,
        budget_bytes=64 * 1024 * 1024,
        ratio=0.01,
        prefetch_depth=1,
    )


class TestHaloExpansion:
    """Tests for halo window expansion utilities."""

    def test_expand_window_interior(self):
        """Interior window expands by full halo in all directions."""
        halo = 2
        height, width = 100, 100
        y0, x0, y1, x1 = 20, 20, 40, 40

        halo_y0, halo_x0, halo_y1, halo_x1, top_pad, left_pad, bottom_pad, right_pad = \
            _expand_window_for_halo(y0, x0, y1, x1, halo, height, width)

        assert halo_y0 == 18
        assert halo_x0 == 18
        assert halo_y1 == 42
        assert halo_x1 == 42
        assert top_pad == 2
        assert left_pad == 2
        assert bottom_pad == 2
        assert right_pad == 2

    def test_expand_window_top_left_edge(self):
        """Edge window at top-left is clamped to raster bounds."""
        halo = 3
        height, width = 100, 100
        y0, x0, y1, x1 = 0, 0, 20, 20

        halo_y0, halo_x0, halo_y1, halo_x1, top_pad, left_pad, _, _ = \
            _expand_window_for_halo(y0, x0, y1, x1, halo, height, width)

        assert halo_y0 == 0  # Clamped to 0
        assert halo_x0 == 0
        assert top_pad == 0  # No halo at top edge
        assert left_pad == 0

    def test_expand_window_bottom_right_edge(self):
        """Edge window at bottom-right is clamped to raster bounds."""
        halo = 3
        height, width = 100, 100
        y0, x0, y1, x1 = 80, 80, 100, 100

        halo_y0, halo_x0, halo_y1, halo_x1, _, _, bottom_pad, right_pad = \
            _expand_window_for_halo(y0, x0, y1, x1, halo, height, width)

        assert halo_y1 == 100  # Clamped to height
        assert halo_x1 == 100
        assert bottom_pad == 0  # No halo at bottom edge
        assert right_pad == 0


class TestCropHalo:
    """Tests for cropping halo from result arrays."""

    def test_crop_2d_array(self):
        """Crop halo from 2D array."""
        result = np.arange(36).reshape(6, 6)
        cropped = _crop_halo_from_result(result, top_pad=1, left_pad=1, core_height=4, core_width=4)

        assert cropped.shape == (4, 4)
        expected = result[1:5, 1:5]
        np.testing.assert_array_equal(cropped, expected)

    def test_crop_3d_array(self):
        """Crop halo from 3D (channels, height, width) array."""
        result = np.arange(2 * 6 * 6).reshape(2, 6, 6)
        cropped = _crop_halo_from_result(result, top_pad=1, left_pad=2, core_height=3, core_width=3)

        assert cropped.shape == (2, 3, 3)
        expected = result[:, 1:4, 2:5]
        np.testing.assert_array_equal(cropped, expected)

    def test_crop_with_zero_padding(self):
        """Zero padding returns the full array."""
        result = np.arange(16).reshape(4, 4)
        cropped = _crop_halo_from_result(result, top_pad=0, left_pad=0, core_height=4, core_width=4)

        np.testing.assert_array_equal(cropped, result)


class TestGlobalBlockAlignment:
    """Tests for global block grid alignment."""

    def test_block_grid_shape_no_padding(self):
        """Block grid shape when dimensions are exact multiples of resolution."""
        height, width, resolution = 64, 128, 16
        blocks_h, blocks_w, h_pad, w_pad = _block_grid_shape(height, width, resolution)

        assert blocks_h == 4
        assert blocks_w == 8
        assert h_pad == (0, 0)
        assert w_pad == (0, 0)

    def test_block_grid_shape_with_padding(self):
        """Block grid shape when dimensions need padding."""
        height, width, resolution = 65, 130, 16
        blocks_h, blocks_w, h_pad, w_pad = _block_grid_shape(height, width, resolution)

        assert blocks_h == 5
        assert blocks_w == 9
        assert h_pad[1] == 15  # Pad to 80
        assert w_pad[1] == 14  # Pad to 144

    def test_globally_aligned_descriptors_position(self):
        """Verify that global block indices are computed correctly."""
        # Create a synthetic array
        array = np.random.randn(3, 48, 48).astype(np.float32)
        resolution = 16
        device = torch.device("cpu")
        compute_dtype = torch.float32
        halo = DESCRIPTOR_HALO_PIXELS

        # Simulate a chunk at global position (16, 16) to (32, 32) with halo
        global_y0, global_x0 = 16, 16

        # The descriptor function should return global block indices
        descriptors, block_shape, gb_y0, gb_x0, n_blocks_h, n_blocks_w = \
            _compute_globally_aligned_descriptors(
                array,
                resolution,
                global_y0,
                global_x0,
                64,  # global height
                64,  # global width
                device,
                compute_dtype,
                halo=halo,
            )

        # Global block (0,0) corresponds to pixels (0-15, 0-15)
        # Global block (1,1) corresponds to pixels (16-31, 16-31)
        assert gb_y0 == 1  # y=16 // 16 = 1
        assert gb_x0 == 1  # x=16 // 16 = 1


class TestSeamPrevention:
    """End-to-end tests that verify seam prevention."""

    def test_single_chunk_vs_multi_chunk_consistency(self, monkeypatch):
        """Labels should match whether processed in one chunk or multiple chunks."""
        # Use deterministic centers
        def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
            np.random.seed(42)
            # Generate reproducible centers
            return np.random.randn(num_clusters, data_np.shape[1]).astype(np.float32)

        monkeypatch.setattr(kmeans_runtime, "_torch_kmeans", fake_torch_kmeans)

        # Create a synthetic gradient image that would reveal seams
        size = 64
        array = np.zeros((3, size, size), dtype=np.float32)
        for c in range(3):
            array[c] = np.linspace(0, 255, size * size).reshape(size, size)

        # Single chunk processing (no boundaries)
        labels_single = funcs.execute_kmeans_segmentation(
            array.copy(),
            num_segments=4,
            resolution=8,
            chunk_plan=_chunk_plan(chunk_size=128),  # Large enough to avoid chunking
            device_hint=torch.device("cpu"),
        )

        # Multi-chunk processing (forced boundaries)
        labels_multi = funcs.execute_kmeans_segmentation(
            array.copy(),
            num_segments=4,
            resolution=8,
            chunk_plan=_chunk_plan(chunk_size=32),  # Force multiple chunks
            device_hint=torch.device("cpu"),
        )

        # Labels should be identical or very close (boundary pixels may differ slightly)
        match_ratio = np.mean(labels_single == labels_multi)

        # With proper halo and alignment, we expect >95% match
        assert match_ratio >= 0.95, f"Match ratio {match_ratio:.2%} below 95% threshold"

    def test_boundary_labels_stable(self, monkeypatch):
        """Labels at chunk boundaries should be stable."""
        def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
            # Fixed centers based on RGB brightness
            return np.array([
                [50.0, 50.0, 50.0],
                [200.0, 200.0, 200.0],
            ], dtype=np.float32)

        monkeypatch.setattr(kmeans_runtime, "_torch_kmeans", fake_torch_kmeans)

        # Create image with clear color boundaries
        size = 64
        array = np.zeros((3, size, size), dtype=np.float32)
        # Left half dark, right half bright
        array[:, :, :size//2] = 60.0
        array[:, :, size//2:] = 190.0

        labels = funcs.execute_kmeans_segmentation(
            array,
            num_segments=2,
            resolution=8,
            chunk_plan=_chunk_plan(chunk_size=32),  # Chunk boundary at x=32
            device_hint=torch.device("cpu"),
        )

        # The chunk boundary is at x=32
        # Labels should be consistent across the boundary
        left_boundary = labels[:, 28:32]
        right_boundary = labels[:, 32:36]

        # Left side should be one label, right side should be another
        # (they won't be equal, but each side should be internally consistent)
        left_unique = np.unique(left_boundary)
        right_unique = np.unique(right_boundary)

        # Each side should have predominantly one label
        assert len(left_unique) <= 2, "Left boundary has too many labels"
        assert len(right_unique) <= 2, "Right boundary has too many labels"

    def test_no_visible_seam_pattern(self, monkeypatch):
        """No systematic seam artifacts at chunk boundaries."""
        def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
            np.random.seed(123)
            return np.random.randn(num_clusters, data_np.shape[1]).astype(np.float32) * 50 + 128

        monkeypatch.setattr(kmeans_runtime, "_torch_kmeans", fake_torch_kmeans)

        # Uniform image - should produce uniform labels
        size = 64
        array = np.full((3, size, size), 128.0, dtype=np.float32)

        labels = funcs.execute_kmeans_segmentation(
            array,
            num_segments=3,
            resolution=8,
            chunk_plan=_chunk_plan(chunk_size=32),
            device_hint=torch.device("cpu"),
        )

        # For a uniform image, we expect mostly uniform labels
        # Any seam would show as a different label at chunk boundaries
        unique_labels = np.unique(labels)

        # With uniform input and proper halo/alignment, should have only 1 label
        # (or at most 2 if there's some numerical precision issue)
        assert len(unique_labels) <= 2, f"Too many labels for uniform image: {unique_labels}"


class TestHaloPixelValue:
    """Tests verifying the halo pixel configuration."""

    def test_descriptor_halo_covers_smoothing_kernel(self):
        """DESCRIPTOR_HALO_PIXELS should provide margin beyond the 3x3 kernel."""
        # 3 pixels eliminates all chunk boundary artifacts
        assert DESCRIPTOR_HALO_PIXELS >= 3

    def test_default_halo_matches_descriptor_halo(self):
        """DEFAULT_HALO_PIXELS in chunking should be consistent."""
        assert DEFAULT_HALO_PIXELS >= DESCRIPTOR_HALO_PIXELS


class TestGlobalScaling:
    """Tests verifying that feature scaling is consistent (not per-chunk)."""

    def test_no_per_chunk_normalization(self, monkeypatch):
        """Verify no per-chunk feature normalization is applied."""
        # Track if any normalization function is called
        normalize_calls = []

        original_pool = kmeans_runtime._smooth_and_pool_descriptors

        def tracking_pool(*args, **kwargs):
            result = original_pool(*args, **kwargs)
            descriptors = result[0]
            # Check that descriptors are not normalized (mean != 0 or std != 1)
            normalize_calls.append({
                "mean": np.mean(descriptors),
                "std": np.std(descriptors),
            })
            return result

        monkeypatch.setattr(kmeans_runtime, "_smooth_and_pool_descriptors", tracking_pool)

        # Run segmentation
        array = np.random.randn(3, 64, 64).astype(np.float32) * 50 + 100

        funcs.execute_kmeans_segmentation(
            array,
            num_segments=3,
            resolution=16,
            chunk_plan=None,  # Use default
            device_hint=torch.device("cpu"),
        )

        # Verify descriptors are not z-score normalized
        for call in normalize_calls:
            # If normalized, mean would be ~0 and std would be ~1
            # Original descriptors should have different stats
            assert call["mean"] != pytest.approx(0.0, abs=0.5) or call["std"] != pytest.approx(1.0, abs=0.2)


class TestBlockOverlap:
    """Tests for block-level overlap stitching."""

    def test_block_overlap_constant_exists(self):
        """BLOCK_OVERLAP constant should be defined and >= 1."""
        assert BLOCK_OVERLAP >= 1

    def test_overlap_prevents_1px_grid(self, monkeypatch):
        """Overlap stitching should prevent visible 1px grid at chunk boundaries.

        This test compares chunked vs non-chunked processing. If overlap stitching
        works correctly, the results should be nearly identical.
        """
        def fake_torch_kmeans(data_np, num_clusters, device=None, compute_dtype=None, seed=None):
            np.random.seed(42)
            return np.random.randn(num_clusters, data_np.shape[1]).astype(np.float32) * 30 + 100

        monkeypatch.setattr(kmeans_runtime, "_torch_kmeans", fake_torch_kmeans)

        # Create gradient image that would reveal seams
        size = 128
        array = np.zeros((3, size, size), dtype=np.float32)
        for c in range(3):
            array[c] = np.linspace(0, 200, size * size).reshape(size, size)

        # Process with single chunk (no boundaries)
        labels_single = funcs.execute_kmeans_segmentation(
            array.copy(),
            num_segments=4,
            resolution=8,
            chunk_plan=_chunk_plan(chunk_size=256),  # Large enough for single chunk
            device_hint=torch.device("cpu"),
        )

        # Process with multiple chunks (has boundaries)
        labels_multi = funcs.execute_kmeans_segmentation(
            array.copy(),
            num_segments=4,
            resolution=8,
            chunk_plan=_chunk_plan(chunk_size=32),  # Forces multiple chunks
            device_hint=torch.device("cpu"),
        )

        # With proper overlap stitching, results should be very similar
        match_ratio = np.mean(labels_single == labels_multi)
        assert match_ratio >= 0.90, f"Match ratio {match_ratio:.2%} indicates seam artifacts"

    def test_chunk_iteration_uses_overlap_stride(self, monkeypatch):
        """Verify that chunk iteration stride accounts for overlap."""
        from runtime.kmeans import _block_chunk_plan

        # 64x64 image with resolution 8 = 8x8 blocks
        # chunk_size 32 / resolution 8 = 4 blocks per chunk
        # With BLOCK_OVERLAP=1, stride should be 4-1=3 blocks

        plan = _chunk_plan(chunk_size=32)
        blocks_h, blocks_w, _, _, block_chunk_h, block_chunk_w, y_starts, x_starts = \
            _block_chunk_plan(64, 64, 8, plan)

        # With 8 blocks and chunk of 4, stride of 3:
        # y_starts should cover all blocks with overlapping chunks
        assert blocks_h == 8
        assert blocks_w == 8
        assert block_chunk_h == 4
        assert len(y_starts) >= 2  # Should have multiple chunks
        assert len(x_starts) >= 2

        # Verify chunks overlap (not disjoint)
        if len(y_starts) > 1:
            # Second chunk start should be less than first chunk end
            assert y_starts[1] < y_starts[0] + block_chunk_h


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
