"""
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
"""
from pathlib import Path

import numpy as np
import pytest

from segmenter.funcs import _materialize_raster, _require_tiff_path
from segmenter.model.runtime_numpy import load_runtime_model


def test_validate_three_band_numpy_accepts_rgb():
    array = np.ones((3, 4, 5), dtype=np.uint8)
    materialized = _materialize_raster(array)
    np.testing.assert_array_equal(materialized, array)


def test_validate_three_band_numpy_rejects_wrong_channels():
    array = np.ones((4, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-band"):
        _materialize_raster(array)


def test_require_tiff_path_enforces_extension():
    with pytest.raises(ValueError, match=r"\.tif|\.tiff"):
        _require_tiff_path("foo.jpg")


def test_runtime_artifact_loads_and_predicts():
    root = Path(__file__).resolve().parents[1]
    runtime = load_runtime_model(root / "model" / "best")
    rgb = np.ones((3, 8, 8), dtype=np.float32)
    labels = runtime.predict_labels(rgb, k=3)
    assert labels.shape == (8, 8)
    assert labels.dtype == np.uint8


def test_dependency_specs_cover_numpy_and_optional_torch():
    import dependency_manager

    specs = list(dependency_manager._package_specs())  # pylint: disable=protected-access
    assert specs, "Package spec list must not be empty"

    numpy_specs = [spec for spec in specs if isinstance(spec, dict) and spec.get("import") == "numpy"]
    torch_specs = [spec for spec in specs if isinstance(spec, dict) and spec.get("import") == "torch"]

    assert numpy_specs, "NumPy spec must be present"
    assert torch_specs, "Torch spec should be present for optional acceleration"
    for spec in torch_specs:
        assert spec.get("optional") is True
        assert spec.get("enable_env") == "SEGMENTER_ENABLE_TORCH"


def test_core_runtime_avoids_hard_torch_imports():
    root = Path(__file__).resolve().parents[1]
    runtime_files = [
        root / "funcs.py",
        root / "segmenter.py",
        root / "model" / "runtime_numpy.py",
        root / "qgis_funcs.py",
    ]
    for path in runtime_files:
        text = path.read_text()
        assert "import torch" not in text


def test_packaging_manifest_ships_model_artifacts():
    root = Path(__file__).resolve().parents[1]
    cfg = (root / "pb_tool.cfg").read_text().lower()
    assert "extra_dirs:" in cfg
    assert "model" in cfg.split("extra_dirs:")[-1], "pb_tool configuration must ship the model directory"
    assert (root / "model" / "best" / "model.npz").exists()
    assert (root / "model" / "best" / "meta.json").exists()
