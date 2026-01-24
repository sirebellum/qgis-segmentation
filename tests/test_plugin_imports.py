# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Tests for plugin import structure and K-Means-only runtime."""

from pathlib import Path

import pytest


def test_package_imports_without_qgis():
    pytest.skip("Plugin entrypoint requires QGIS; importability is covered in integration environment.")


def test_runtime_uses_relative_imports_only():
    root = Path(__file__).resolve().parents[1]
    runtime_files = [
        root / "__init__.py",
        root / "segmenter.py",
        root / "funcs.py",
        root / "qgis_funcs.py",
        root / "dependency_manager.py",
        root / "raster_utils.py",
        root / "runtime" / "__init__.py",
        root / "runtime" / "pipeline.py",
        root / "runtime" / "kmeans.py",
    ]

    patterns = [
        "import funcs",
        "from funcs import",
        "import qgis_funcs",
        "from qgis_funcs import",
    ]

    offenders = []
    for path in runtime_files:
        if not path.exists():
            continue
        text = path.read_text()
        for pattern in patterns:
            if pattern in text:
                offenders.append((path, pattern))

    assert not offenders, f"Found absolute imports that should be relative: {offenders}"


def test_segmenter_imports_only_kmeans():
    """Verify segmenter only imports K-Means segmentation function."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "segmenter.py").read_text(encoding="utf-8")
    
    # Should import K-Means
    assert "execute_kmeans_segmentation" in text
    # Should NOT import CNN
    assert "execute_cnn_segmentation" not in text
