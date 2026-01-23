# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Tests to verify K-Means-only runtime pipeline (no CNN)."""

from pathlib import Path


def test_segmenter_routes_to_kmeans_only():
    """Verify segmenter.py uses execute_kmeans_segmentation and does not import CNN functions."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "segmenter.py").read_text(encoding="utf-8")
    assert "execute_kmeans_segmentation" in text
    # CNN paths must NOT be present
    assert "execute_cnn_segmentation" not in text
    assert "load_model" not in text or "self.load_model" not in text
    # No model dropdown
    assert "inputLoadModel" not in text
    assert "render_models" not in text
    # Legacy names should still be absent
    assert "legacy_cnn_segmentation" not in text
    assert "legacy_kmeans_segmentation" not in text


def test_pipeline_exports_kmeans_only():
    """Verify runtime/pipeline.py only exports K-Means functions."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "runtime" / "pipeline.py").read_text(encoding="utf-8")
    assert "execute_kmeans_segmentation" in text
    # CNN function must not be defined or exported
    assert "execute_cnn_segmentation" not in text
    assert "fit_global_cnn_centers" not in text
    assert "predict_cnn" not in text


def test_funcs_facade_no_cnn_exports():
    """Verify funcs.py facade does not export CNN symbols."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "funcs.py").read_text(encoding="utf-8")
    assert "execute_kmeans_segmentation" in text
    # CNN exports must be absent from __all__
    assert '"execute_cnn_segmentation"' not in text
    assert '"predict_cnn"' not in text
    assert '"tile_raster"' not in text
