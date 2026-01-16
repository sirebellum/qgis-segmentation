# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import importlib
from pathlib import Path

import pytest


def test_package_imports_without_qgis():
    pkg = importlib.import_module("segmenter")
    mod = importlib.import_module("segmenter.segmenter")

    assert pkg is not None
    assert hasattr(mod, "Segmenter")

    if getattr(mod, "_HAS_QGIS", False):
        pytest.skip("QGIS bindings are available; stub constructor check not applicable.")

    with pytest.raises(ImportError):
        mod.Segmenter(None)


def test_runtime_uses_relative_imports_only():
    root = Path(__file__).resolve().parents[1]
    runtime_files = [
        root / "__init__.py",
        root / "segmenter.py",
        root / "funcs.py",
        root / "qgis_funcs.py",
        root / "dependency_manager.py",
        root / "perf_tuner.py",
        root / "raster_utils.py",
        root / "model" / "__init__.py",
        root / "model" / "runtime_backend.py",
        root / "model" / "runtime_numpy.py",
        root / "model" / "runtime_torch.py",
    ]

    patterns = [
        "import funcs",
        "from funcs import",
        "import model",
        "from model import",
        "import qgis_funcs",
        "from qgis_funcs import",
    ]

    offenders = []
    for path in runtime_files:
        text = path.read_text()
        for pattern in patterns:
            if pattern in text:
                offenders.append((path, pattern))

    assert not offenders, f"Found absolute imports that should be relative: {offenders}"
