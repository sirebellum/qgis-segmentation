# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import math

from scripts.data._gdal_utils import (
    bounds_from_info,
    derive_utm_epsg,
    geotransform_from_info,
    geotransforms_equal,
    pixel_size_from_info,
)


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
