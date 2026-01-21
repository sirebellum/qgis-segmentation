# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import importlib
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
        if not path.exists():
            continue
        text = path.read_text()
        for pattern in patterns:
            if pattern in text:
                offenders.append((path, pattern))

    assert not offenders, f"Found absolute imports that should be relative: {offenders}"
