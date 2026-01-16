# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import os
import pytest


if os.environ.get("QGIS_TESTS") != "1":  # pragma: no cover - optional runtime
    pytest.skip("QGIS tests disabled; set QGIS_TESTS=1 to enable.", allow_module_level=True)

try:
    from qgis.core import QgsApplication  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytest.skip("QGIS not available in this environment.", allow_module_level=True)


def test_qgis_core_imports():
    # Smoke-test that QGIS core is importable when explicitly enabled.
    assert QgsApplication is not None
