# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Tests for runtime/io.py - raster and model materialization."""

from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from runtime.io import _materialize_model, _materialize_raster


class TestMaterializeRaster:
    def test_numpy_array_passthrough(self):
        array = np.random.rand(3, 64, 64).astype(np.float32)
        result = _materialize_raster(array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, array)

    def test_callable_loader(self):
        def loader():
            return np.random.rand(3, 32, 32).astype(np.float32)

        result = _materialize_raster(loader)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 32, 32)

    def test_file_path_string(self):
        # Use existing test fixture
        fixture_path = Path(__file__).parent / "test_tif.tif"
        if not fixture_path.exists():
            pytest.skip("Test fixture test_tif.tif not found")

        result = _materialize_raster(str(fixture_path))
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3


class TestMaterializeModel:
    def test_model_passthrough(self):
        model = torch.nn.Identity()
        result = _materialize_model(model, torch.device("cpu"))
        assert result is model

    def test_callable_loader(self):
        def loader():
            return torch.nn.Linear(10, 5)

        result = _materialize_model(loader, torch.device("cpu"))
        assert isinstance(result, torch.nn.Linear)

    def test_none_model_raises(self):
        with pytest.raises(RuntimeError):
            _materialize_model(None, torch.device("cpu"))
