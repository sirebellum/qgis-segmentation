# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Tests for runtime/chunking.py - chunk aggregation and processing."""

import numpy as np
import pytest

from runtime.chunking import (
    _ChunkAggregator,
    _compute_chunk_starts,
    _label_to_one_hot,
    _normalize_inference_output,
)


class TestComputeChunkStarts:
    def test_single_chunk_for_small_input(self):
        starts = _compute_chunk_starts(length=64, chunk_size=128, stride=64)
        assert starts == [0]

    def test_multiple_chunks_no_overlap(self):
        starts = _compute_chunk_starts(length=256, chunk_size=64, stride=64)
        assert len(starts) >= 4
        assert 0 in starts

    def test_overlap_reduces_stride(self):
        starts = _compute_chunk_starts(length=128, chunk_size=64, stride=48)
        assert len(starts) >= 2
        # With stride 48, chunks should advance by 48 pixels
        if len(starts) > 1:
            assert starts[1] - starts[0] == 48


class TestLabelToOneHot:
    def test_output_shape(self):
        labels = np.array([[0, 1], [2, 0]], dtype=np.int64)
        one_hot = _label_to_one_hot(labels, num_segments=4)
        assert one_hot.shape == (4, 2, 2)

    def test_correct_encoding(self):
        labels = np.array([[0, 1], [1, 0]], dtype=np.int64)
        one_hot = _label_to_one_hot(labels, num_segments=2)
        # Check class 0 channel
        expected_0 = np.array([[1, 0], [0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(one_hot[0], expected_0)
        # Check class 1 channel
        expected_1 = np.array([[0, 1], [1, 0]], dtype=np.float32)
        np.testing.assert_array_equal(one_hot[1], expected_1)


class TestNormalizeInferenceOutput:
    def test_passthrough_label_map(self):
        labels = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.uint8)
        result_labels, result_scores = _normalize_inference_output(labels)
        assert result_labels.shape == labels.shape

    def test_dict_output(self):
        labels = np.array([[0, 1], [1, 0]], dtype=np.int32)
        scores = np.random.rand(2, 2, 2).astype(np.float32)
        result_labels, result_scores = _normalize_inference_output({"labels": labels, "scores": scores})
        np.testing.assert_array_equal(result_labels, labels)
        np.testing.assert_array_equal(result_scores, scores)

    def test_tuple_output(self):
        labels = np.array([[0, 1], [1, 0]], dtype=np.int32)
        scores = np.random.rand(2, 2, 2).astype(np.float32)
        result_labels, result_scores = _normalize_inference_output((labels, scores))
        np.testing.assert_array_equal(result_labels, labels)
        np.testing.assert_array_equal(result_scores, scores)


class TestChunkAggregator:
    def test_single_chunk_coverage(self):
        agg = _ChunkAggregator(
            height=64,
            width=64,
            num_segments=3,
            chunk_size=64,
            harmonize_labels=False,
        )

        # Add single chunk covering entire area
        labels = np.random.randint(0, 3, size=(64, 64), dtype=np.int32)
        agg.add(labels, region=(0, 0, 64, 64), chunk_data=None, scores=None)

        result = agg.finalize()
        assert result.shape == (64, 64)
        assert result.max() < 3

    def test_multiple_chunks(self):
        agg = _ChunkAggregator(
            height=64,
            width=64,
            num_segments=2,
            chunk_size=32,
            harmonize_labels=False,
        )

        # Add four 32x32 chunks
        for row in [0, 32]:
            for col in [0, 32]:
                labels = np.random.randint(0, 2, size=(32, 32), dtype=np.int32)
                agg.add(labels, region=(row, col, row + 32, col + 32), chunk_data=None, scores=None)

        result = agg.finalize()
        assert result.shape == (64, 64)
        assert result.dtype == np.uint8
