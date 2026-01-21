# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

from pathlib import Path


def test_segmenter_routes_to_execute_pipeline():
    root = Path(__file__).resolve().parents[1]
    text = (root / "segmenter.py").read_text(encoding="utf-8")
    assert "execute_cnn_segmentation" in text
    assert "execute_kmeans_segmentation" in text
    assert "legacy_cnn_segmentation" not in text
    assert "legacy_kmeans_segmentation" not in text
