# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Tests for runtime/latent.py - latent KNN refinement utilities."""

import numpy as np
import pytest
import torch

from runtime.latent import (
    LATENT_KNN_DEFAULTS,
    _latent_knn_soft_refine,
    _resize_label_map,
    _resize_latent_map,
    _stratified_sample_indices,
)


class TestResizeLatentMap:
    def test_resize_matches_target_shape(self):
        latent = np.random.rand(16, 32, 32).astype(np.float32)
        resized = _resize_latent_map(latent, (64, 64))
        assert resized.shape == (16, 64, 64)

    def test_resize_preserves_channels(self):
        latent = np.random.rand(8, 16, 16).astype(np.float32)
        resized = _resize_latent_map(latent, (32, 32))
        assert resized.shape[0] == 8


class TestResizeLabelMap:
    def test_resize_uses_nearest(self):
        # Create a simple label map
        labels = np.zeros((16, 16), dtype=np.int32)
        labels[:8, :] = 1
        labels[8:, :] = 2

        resized = _resize_label_map(labels, (32, 32))
        assert resized.shape == (32, 32)
        # Labels should still be integers
        assert resized.dtype in (np.int32, np.int64)


class TestStratifiedSampleIndices:
    def test_respects_max_points(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        indices = _stratified_sample_indices(labels, max_points=6)
        assert indices is None or len(indices) <= 6

    def test_samples_from_all_classes(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        indices = _stratified_sample_indices(labels, max_points=9)
        if indices is not None:
            sampled_labels = labels[indices]
            assert len(np.unique(sampled_labels)) == 3

    def test_returns_none_when_under_limit(self):
        labels = np.array([0, 1, 2])
        indices = _stratified_sample_indices(labels, max_points=100)
        assert indices is None  # Returns None when no sampling needed


class TestLatentKNNDefaults:
    def test_defaults_are_reasonable(self):
        assert LATENT_KNN_DEFAULTS["mix"] > 0
        assert LATENT_KNN_DEFAULTS["mix"] <= 1
        assert LATENT_KNN_DEFAULTS["neighbors"] > 0
        assert LATENT_KNN_DEFAULTS["temperature"] > 0
        assert LATENT_KNN_DEFAULTS["iterations"] >= 1


class TestLatentKNNSoftRefine:
    def test_output_shape_matches_input(self):
        labels = np.random.randint(0, 4, size=(64, 64), dtype=np.int32)
        latent = np.random.rand(16, 64, 64).astype(np.float32)
        centers = np.random.rand(4, 16).astype(np.float32)

        refined = _latent_knn_soft_refine(
            latent_map=latent,
            lowres_labels=labels,
            centers=centers,
            num_segments=4,
            config={"enabled": True, "iterations": 1, "neighbors": 4},
            status_callback=None,
            cancel_token=None,
        )

        assert refined.shape == labels.shape

    def test_disabled_returns_labels(self):
        labels = np.random.randint(0, 3, size=(32, 32), dtype=np.int32)
        latent = np.random.rand(8, 32, 32).astype(np.float32)
        centers = np.random.rand(3, 8).astype(np.float32)

        refined = _latent_knn_soft_refine(
            latent_map=latent,
            lowres_labels=labels,
            centers=centers,
            num_segments=3,
            config={"enabled": False},
            status_callback=None,
            cancel_token=None,
        )

        assert refined.shape == labels.shape
        assert refined.dtype == np.uint8
