# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""GDAL-backed geotransform helpers used in tests and data scripts."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Tuple, Union

from osgeo import gdal, osr  # type: ignore

GeoTransform = Tuple[float, float, float, float, float, float]
_InfoLike = Union[dict, str, Path, "gdal.Dataset"]


def _open_dataset(info: _InfoLike):
    if isinstance(info, (str, Path)):
        dataset = gdal.Open(str(info))
        if dataset is None:
            raise RuntimeError(f"Unable to open GDAL dataset: {info}")
        return dataset
    if hasattr(info, "GetGeoTransform") and hasattr(info, "RasterXSize"):
        return info  # assume GDAL dataset-like object
    return None


def _extract_geoinfo(info: _InfoLike):
    dataset = _open_dataset(info)
    if dataset is not None:
        geo = dataset.GetGeoTransform()
        size = (dataset.RasterXSize, dataset.RasterYSize)
    elif isinstance(info, dict):
        geo = info.get("geoTransform")
        size = info.get("size")
    else:
        raise TypeError("info must be a GDAL dataset, path, or info dict")

    if not geo or len(geo) != 6:
        raise ValueError("geoTransform must have 6 elements")
    if not size or len(size) != 2:
        raise ValueError("size must be [width, height]")

    return tuple(float(v) for v in geo), (int(size[0]), int(size[1]))


def geotransform_from_info(info: _InfoLike) -> GeoTransform:
    geo, _ = _extract_geoinfo(info)
    return geo


def pixel_size_from_info(info: _InfoLike) -> Tuple[float, float]:
    gt = geotransform_from_info(info)
    return float(gt[1]), float(gt[5])


def bounds_from_info(info: _InfoLike) -> Tuple[float, float, float, float]:
    gt, size = _extract_geoinfo(info)
    width, height = size
    x_min = float(gt[0])
    y_max = float(gt[3])
    x_max = x_min + width * float(gt[1])
    y_min = y_max + height * float(gt[5])
    return x_min, y_min, x_max, y_max


def geotransforms_equal(a: Sequence[float], b: Sequence[float], tol: float = 0.0) -> bool:
    if len(a) != 6 or len(b) != 6:
        return False
    for va, vb in zip(a, b):
        if math.isfinite(tol) and tol > 0.0:
            if abs(float(va) - float(vb)) > tol:
                return False
        else:
            if float(va) != float(vb):
                return False
    return True


def derive_utm_epsg(lon: float, lat: float) -> int:
    if not math.isfinite(lon) or not math.isfinite(lat):
        raise ValueError("Longitude and latitude must be finite numbers")
    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(zone, 60))
    north = lat >= 0
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    srs.SetUTM(zone, north)
    code = srs.GetAuthorityCode("PROJCS")
    if code:
        try:
            return int(code)
        except ValueError:
            pass
    return 32600 + zone if north else 32700 + zone


__all__ = [
    "bounds_from_info",
    "derive_utm_epsg",
    "geotransform_from_info",
    "geotransforms_equal",
    "pixel_size_from_info",
]
