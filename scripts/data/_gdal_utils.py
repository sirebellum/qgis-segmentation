# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""GDAL CLI helpers for dataset preparation scripts.

These wrappers favor subprocess calls to avoid adding heavy Python GDAL
bindings; failures should be actionable (install gdalinfo/gdalwarp).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, List, Optional


def require_gdal_tools(tools: Iterable[str] = ("gdalinfo", "gdalwarp", "gdal_translate", "gdalbuildvrt")) -> None:
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing GDAL CLI tools: {joined}. Install GDAL (e.g., 'brew install gdal' or 'apt-get install gdal-bin')."
        )


def run_cmd(cmd: List[str], logger: Optional[Callable[[str], None]] = None, cwd: Optional[Path] = None) -> str:
    if logger:
        logger("$ " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no stderr)"
        stdout = result.stdout.strip()
        raise RuntimeError(f"Command failed ({cmd[0]}): {stderr}\n{stdout}")
    return result.stdout


def gdalinfo_json(path: Path) -> dict:
    out = run_cmd(["gdalinfo", "-json", str(path)])
    return json.loads(out)


def derive_utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def geotransform_from_info(info: dict) -> tuple:
    geo = info.get("geoTransform")
    if not geo or len(geo) != 6:
        raise ValueError("Invalid gdalinfo geoTransform")
    return tuple(geo)


def bounds_from_info(info: dict) -> tuple:
    geo = geotransform_from_info(info)
    width, height = info.get("size", [0, 0])
    minx = geo[0]
    maxy = geo[3]
    px_w = geo[1]
    px_h = geo[5]
    maxx = minx + width * px_w
    miny = maxy + height * px_h
    return (minx, miny, maxx, maxy)


def pixel_size_from_info(info: dict) -> tuple:
    geo = geotransform_from_info(info)
    return (geo[1], geo[5])


def geotransforms_equal(gt1: Iterable[float], gt2: Iterable[float], tol: float = 1e-6) -> bool:
    values1 = list(gt1)
    values2 = list(gt2)
    if len(values1) != len(values2):
        return False
    for a, b in zip(values1, values2):
        if abs(a - b) > tol:
            return False
    return True
