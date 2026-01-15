# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""GDAL CLI helpers for dataset preparation scripts.

These wrappers favor subprocess calls to avoid adding heavy Python GDAL
bindings; failures should be actionable (install gdalinfo/gdalwarp).

Added in Phase 7:
- Optional env overrides so Requester Pays buckets (e.g., NAIP on AWS) work.
- Convenience wrappers for VRT build and warping to a fixed grid.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


def require_gdal_tools(tools: Iterable[str] = ("gdalinfo", "gdalwarp", "gdal_translate", "gdalbuildvrt", "ogr2ogr")) -> None:
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing GDAL CLI tools: {joined}. Install GDAL (e.g., 'brew install gdal' or 'apt-get install gdal-bin')."
        )


def run_cmd(cmd: List[str], logger: Optional[Callable[[str], None]] = None, cwd: Optional[Path] = None, env: Optional[dict] = None) -> str:
    if logger:
        logger("$ " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
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


# Grid helpers
class GridSpec:
    """Target grid definition for warps and validation."""

    def __init__(self, epsg: int, pixel_size: Tuple[float, float], bounds: Tuple[float, float, float, float], size: Tuple[int, int]):
        self.epsg = epsg
        self.pixel_size = pixel_size
        self.bounds = bounds
        self.size = size

    def as_warp_args(self) -> List[str]:
        minx, miny, maxx, maxy = self.bounds
        px_x, px_y = self.pixel_size
        width, height = self.size
        return [
            "-t_srs",
            f"EPSG:{self.epsg}",
            "-tr",
            str(px_x),
            str(abs(px_y)),
            "-te",
            str(minx),
            str(miny),
            str(maxx),
            str(maxy),
            "-ts",
            str(width),
            str(height),
            "-tap",
        ]


def build_vrt(sources: Sequence[str], vrt_path: Path, resampling: str = "bilinear", env: Optional[dict] = None, logger: Optional[Callable[[str], None]] = None) -> None:
    if not sources:
        raise ValueError("At least one source is required to build VRT")
    run_cmd([
        "gdalbuildvrt",
        "-r",
        resampling,
        str(vrt_path),
        *sources,
    ], logger=logger, env=env)


def warp_to_grid(src: Path, dst: Path, grid: GridSpec, resampling: str = "bilinear", dst_nodata: Optional[float] = None, extra_args: Optional[List[str]] = None, env: Optional[dict] = None, logger: Optional[Callable[[str], None]] = None) -> str:
    args = [
        "gdalwarp",
        str(src),
        str(dst),
        "-r",
        resampling,
        "-overwrite",
    ]
    if dst_nodata is not None:
        args.extend(["-dstnodata", str(dst_nodata)])
    args.extend(grid.as_warp_args())
    if extra_args:
        args.extend(extra_args)
    return run_cmd(args, logger=logger, env=env)


def subset_vector_by_bbox(vector_path: Path, bbox: Tuple[float, float, float, float], out_path: Optional[Path] = None, logger: Optional[Callable[[str], None]] = None) -> Path:
    """Clip a vector index by bbox (WGS84) using ogr2ogr.

    Returns the path to the subset GeoJSON (temp if not provided).
    """

    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix="naip_index_subset_", suffix=".geojson", delete=False)
        out_path = Path(tmp.name)
        tmp.close()
    minx, miny, maxx, maxy = bbox
    run_cmd(
        [
            "ogr2ogr",
            "-f",
            "GeoJSON",
            str(out_path),
            str(vector_path),
            "-spat",
            str(minx),
            str(miny),
            str(maxx),
            str(maxy),
            "-t_srs",
            "EPSG:4326",
        ],
        logger=logger,
    )
    return out_path
