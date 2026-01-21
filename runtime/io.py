# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Raster/model materialization helpers."""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

try:
    from osgeo import gdal  # type: ignore
except ImportError:  # pragma: no cover - requires QGIS runtime
    gdal = None  # type: ignore


def _materialize_raster(raster_input):
    if isinstance(raster_input, np.ndarray):
        return raster_input
    if callable(raster_input):
        materialized = raster_input()
        if not isinstance(materialized, np.ndarray):
            raise TypeError("Raster loader must return a numpy.ndarray")
        return materialized
    if isinstance(raster_input, str):
        source = raster_input.split("|")[0]
        if gdal is None:
            raise RuntimeError("GDAL is required to read raster sources.")
        dataset = gdal.Open(source)
        if dataset is None:
            raise RuntimeError(f"Unable to open raster source: {source}")
        array = dataset.ReadAsArray()
        dataset = None
        if array is None:
            raise RuntimeError("Raster source returned no data.")
        return np.ascontiguousarray(array)
    raise TypeError(f"Unsupported raster input type: {type(raster_input)!r}")


def _materialize_model(model_or_loader: Any, device: Optional[Any]):
    if callable(model_or_loader) and not hasattr(model_or_loader, "forward"):
        model: Any = model_or_loader()
    else:
        model = model_or_loader
    if model is None:
        raise RuntimeError("CNN model provider returned no model instance.")
    to_fn = getattr(model, "to", None)
    if callable(to_fn) and device is not None:
        try:
            model = to_fn(device)
        except Exception:  # pragma: no cover - best effort device placement
            pass
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        try:
            model = eval_fn()
        except Exception:
            pass
    return model
