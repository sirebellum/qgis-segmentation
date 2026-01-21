# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import math

import pytest

try:
    from scripts.data._gdal_utils import (
        bounds_from_info,
        derive_utm_epsg,
        geotransform_from_info,
        geotransforms_equal,
        pixel_size,
        pixel_size_from_info,
        raster_bounds,
        raster_center,
        resolve_intersection,
    )
except ModuleNotFoundError:
    pytest.skip("scripts package not available", allow_module_level=True)


def test_geotransform_helpers_are_consistent():
    info = {"geoTransform": [10.0, 2.0, 0.0, 20.0, 0.0, -2.0], "size": [3, 2]}
    gt = geotransform_from_info(info)

    assert pixel_size_from_info(info) == (2.0, -2.0)
    assert bounds_from_info(info) == (10.0, 16.0, 16.0, 20.0)
    assert geotransforms_equal(gt, gt, tol=1e-9)
    assert not geotransforms_equal(gt, (10.0, 2.0, 0.0, 20.0, 0.0, -2.1), tol=1e-9)

    epsg = derive_utm_epsg(-105.0, 40.0)
    assert epsg == 32613
    assert math.isfinite(epsg)


def test_raster_bounds_center_and_intersection():
    gt = (10.0, 2.0, 0.0, 20.0, 0.0, -2.0)
    bounds = raster_bounds(gt, 3, 2)
    assert bounds == (10.0, 16.0, 16.0, 20.0)
    assert raster_center(bounds) == (13.0, 18.0)
    assert pixel_size(gt) == (2.0, -2.0)

    intersection = resolve_intersection((0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0))
    assert intersection == (1.0, 1.0, 2.0, 2.0)
