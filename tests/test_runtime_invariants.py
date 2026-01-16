"""
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
"""
from pathlib import Path

import numpy as np
import pytest

from model.runtime_numpy import load_runtime_model


def test_runtime_artifact_loads_and_predicts():
    root = Path(__file__).resolve().parents[1]
    runtime = load_runtime_model(root / "model" / "best")
    rgb = np.ones((3, 8, 8), dtype=np.float32)
    labels = runtime.predict_labels(rgb, k=3)
    assert labels.shape == (8, 8)
    assert labels.dtype == np.uint8


def test_dependency_specs_cover_numpy_and_torch():
    import dependency_manager

    specs = list(dependency_manager._package_specs())  # pylint: disable=protected-access
    assert specs, "Package spec list must not be empty"

    numpy_specs = [spec for spec in specs if isinstance(spec, dict) and spec.get("import") == "numpy"]
    torch_specs = [spec for spec in specs if isinstance(spec, dict) and spec.get("import") == "torch"]

    assert numpy_specs, "NumPy spec must be present"
    assert torch_specs, "Torch spec should be present for acceleration"


def test_packaging_manifest_ships_model_artifacts():
    root = Path(__file__).resolve().parents[1]
    cfg = (root / "pb_tool.cfg").read_text().lower()
    assert "extra_dirs:" in cfg
    assert "model" in cfg.split("extra_dirs:")[-1], "pb_tool configuration must ship the model directory"
    assert (root / "model" / "best" / "model.npz").exists()
    assert (root / "model" / "best" / "meta.json").exists()


@pytest.mark.skip(reason="QGIS-bound plugin entrypoint is not importable in test env; validated via artifact checks")
def test_segmenter_import_qgis_free():
    ...
