# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Prepare NAIP RGB + USGS 3DEP DEM tiles for training.

This script downloads NAIP imagery and 3DEP DEM for fixed city AOIs plus a
seeded mountainous AOI, reprojects both to a shared UTM grid, aligns DEM to the
RGB grid, tiles into patches, and emits a manifest.jsonl. Heavy work is handled
by GDAL CLI tools; the script stays dependency-light and offers a dry-run and
validation-only mode.

Notes:
- Endpoint logic is isolated in _usgs_tnm_provider so it can be swapped easily.
- DEM selection prefers 1 m; falls back to 1/3 arc-second then 1 arc-second.
- Alignment is strict: DEM is warped to the NAIP processed grid with -tap and
  matched resolution/bounds/size.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ._gdal_utils import (
    bounds_from_info,
    derive_utm_epsg,
    gdalinfo_json,
    geotransform_from_info,
    geotransforms_equal,
    pixel_size_from_info,
    require_gdal_tools,
    run_cmd,
)
from ._usgs_tnm_provider import (
    DEM_DATASETS,
    NAIP_DATASET,
    Product,
    download_product,
    query_products,
    select_best_dem,
    select_best_naip,
)

# AOIs: fixed city footprints (lon/lat in EPSG:4326)
CITY_AOIS: Dict[str, Tuple[float, float, float, float]] = {
    "chicago_il": (-87.72, 41.84, -87.60, 41.92),
    "phoenix_az": (-112.15, 33.39, -111.95, 33.55),
    "atlanta_ga": (-84.45, 33.70, -84.34, 33.82),
}

# Mountainous sampling region (Rockies, CO)
MOUNTAIN_REGION = (-106.4, 39.2, -105.6, 39.8)


@dataclass
class ManifestRecord:
    aoi_name: str
    tile_id: str
    rgb_path: str
    dem_path: str
    epsg: int
    pixel_size: Tuple[float, float]
    width: int
    height: int
    bounds: Tuple[float, float, float, float]
    geotransform: Tuple[float, float, float, float, float, float]
    source_naip: Optional[str]
    source_dem: Optional[str]
    dem_tier: Optional[str] = None


def _log(msg: str) -> None:
    print(msg, flush=True)


def _ensure_dirs(base: Path) -> Dict[str, Path]:
    layout = {
        "root": base,
        "raw_naip": base / "raw" / "naip",
        "raw_dem": base / "raw" / "dem",
        "proc_rgb": base / "processed" / "rgb",
        "proc_dem": base / "processed" / "dem",
        "manifest": base / "processed" / "manifest.jsonl",
        "logs": base / "logs",
        "cache": base / "cache",
    }
    for key, path in layout.items():
        if key == "manifest":
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    return layout


def _sample_mountain_aoi(seed: int, size_m: int) -> Tuple[str, Tuple[float, float, float, float]]:
    rng = random.Random(seed)
    min_lon, min_lat, max_lon, max_lat = MOUNTAIN_REGION
    for _ in range(100):
        lon = rng.uniform(min_lon, max_lon)
        lat = rng.uniform(min_lat, max_lat)
        half_dlat = (size_m / 2) / 111_320.0
        lat_rad = math.radians(lat)
        half_dlon = half_dlat / max(math.cos(lat_rad), 1e-6)
        aoi = (lon - half_dlon, lat - half_dlat, lon + half_dlon, lat + half_dlat)
        name = f"rockies_{seed}"
        return name, aoi
    raise RuntimeError("Failed to sample mountainous AOI")


def _select_products(aoi_name: str, bbox: Sequence[float], prefer_1m: bool) -> Tuple[Product, Product]:
    naip_products = query_products(NAIP_DATASET, bbox, logger=_log)
    naip = select_best_naip(naip_products)
    if not naip:
        raise RuntimeError(f"No NAIP products found for {aoi_name}")

    dem_candidates: List[Product] = []
    for dataset in DEM_DATASETS:
        dem_candidates.extend(query_products(dataset, bbox, logger=_log))
    dem = select_best_dem(dem_candidates, prefer_1m=prefer_1m)
    if not dem:
        raise RuntimeError(f"No DEM products found for {aoi_name}")
    return naip, dem


def _mosaic_and_clip(inputs: List[Path], bbox: Sequence[float], vrt_path: Path, out_path: Path) -> None:
    run_cmd(["gdalbuildvrt", str(vrt_path), *[str(p) for p in inputs]], logger=_log)
    run_cmd(
        [
            "gdalwarp",
            str(vrt_path),
            str(out_path),
            "-te",
            str(bbox[0]),
            str(bbox[1]),
            str(bbox[2]),
            str(bbox[3]),
            "-t_srs",
            "EPSG:4326",
            "-r",
            "bilinear",
            "-overwrite",
        ],
        logger=_log,
    )


def _reproject_to_utm(src: Path, epsg: int, dst: Path) -> dict:
    run_cmd(
        [
            "gdalwarp",
            str(src),
            str(dst),
            "-t_srs",
            f"EPSG:{epsg}",
            "-r",
            "bilinear",
            "-overwrite",
        ],
        logger=_log,
    )
    return gdalinfo_json(dst)


def _align_dem_to_rgb(dem: Path, rgb_info: dict, epsg: int, dst: Path) -> dict:
    gt = geotransform_from_info(rgb_info)
    width, height = rgb_info.get("size", [0, 0])
    minx, miny, maxx, maxy = bounds_from_info(rgb_info)
    run_cmd(
        [
            "gdalwarp",
            str(dem),
            str(dst),
            "-t_srs",
            f"EPSG:{epsg}",
            "-r",
            "bilinear",
            "-te",
            str(minx),
            str(miny),
            str(maxx),
            str(maxy),
            "-tr",
            str(gt[1]),
            str(abs(gt[5])),
            "-ts",
            str(width),
            str(height),
            "-tap",
            "-overwrite",
        ],
        logger=_log,
    )
    return gdalinfo_json(dst)


def _tile_pair(aoi: str, rgb: Path, dem: Path, out_rgb_dir: Path, out_dem_dir: Path, patch: int, stride: int, epsg: int, manifest: List[ManifestRecord], src_ids: Tuple[str, str]) -> None:
    rgb_info = gdalinfo_json(rgb)
    dem_info = gdalinfo_json(dem)
    if rgb_info.get("coordinateSystem") != dem_info.get("coordinateSystem"):
        raise RuntimeError("CRS mismatch between RGB and DEM after alignment")
    if rgb_info.get("size") != dem_info.get("size"):
        raise RuntimeError("Size mismatch between RGB and DEM after alignment")
    if not geotransforms_equal(rgb_info.get("geoTransform", []), dem_info.get("geoTransform", []), tol=1e-6):
        raise RuntimeError("Geotransform mismatch between RGB and DEM after alignment")

    width, height = rgb_info["size"]
    px_size = pixel_size_from_info(rgb_info)
    base_gt = geotransform_from_info(rgb_info)
    minx, miny, maxx, maxy = bounds_from_info(rgb_info)

    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    out_dem_dir.mkdir(parents=True, exist_ok=True)

    x = 0
    while x + patch <= width:
        y = 0
        while y + patch <= height:
            tile_id = f"{aoi}_x{x}_y{y}"
            rgb_tile = out_rgb_dir / f"{tile_id}.tif"
            dem_tile = out_dem_dir / f"{tile_id}.tif"
            run_cmd([
                "gdal_translate",
                str(rgb),
                str(rgb_tile),
                "-srcwin",
                str(x),
                str(y),
                str(patch),
                str(patch),
                "-co",
                "COMPRESS=LZW",
            ], logger=_log)
            run_cmd([
                "gdal_translate",
                str(dem),
                str(dem_tile),
                "-srcwin",
                str(x),
                str(y),
                str(patch),
                str(patch),
                "-co",
                "COMPRESS=LZW",
            ], logger=_log)
            tile_minx = base_gt[0] + x * base_gt[1]
            tile_maxy = base_gt[3] + y * base_gt[5]
            tile_maxx = tile_minx + patch * base_gt[1]
            tile_miny = tile_maxy + patch * base_gt[5]
            manifest.append(
                ManifestRecord(
                    aoi_name=aoi,
                    tile_id=tile_id,
                    rgb_path=str(rgb_tile),
                    dem_path=str(dem_tile),
                    epsg=epsg,
                    pixel_size=px_size,
                    width=patch,
                    height=patch,
                    bounds=(tile_minx, tile_miny, tile_maxx, tile_maxy),
                    geotransform=base_gt,
                    source_naip=src_ids[0],
                    source_dem=src_ids[1],
                )
            )
            y += stride
        x += stride

    _log(f"Tiled {aoi}: {len(manifest)} total tiles so far")


def _write_manifest(manifest: List[ManifestRecord], manifest_path: Path, root: Path) -> None:
    anchor = root.parent.parent if root.parent.parent.exists() else root
    with manifest_path.open("w") as f:
        for rec in manifest:
            record = asdict(rec)
            record["rgb_path"] = str(Path(record["rgb_path"]).resolve().relative_to(anchor))
            record["dem_path"] = str(Path(record["dem_path"]).resolve().relative_to(anchor))
            f.write(json.dumps(record) + "\n")
    _log(f"Wrote manifest with {len(manifest)} entries: {manifest_path}")


def _validate_only(manifest_path: Path) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    anchor = manifest_path.parent.parent.parent if manifest_path.parent.parent.parent.exists() else manifest_path.parent
    count = 0
    with manifest_path.open() as f:
        for line in f:
            rec = json.loads(line)
            rgb = Path(anchor / rec["rgb_path"]).resolve()
            dem = Path(anchor / rec["dem_path"]).resolve()
            if not rgb.exists() or not dem.exists():
                raise FileNotFoundError(f"Missing tile paths: {rgb} / {dem}")
            rgb_info = gdalinfo_json(rgb)
            dem_info = gdalinfo_json(dem)
            if rgb_info.get("size") != dem_info.get("size"):
                raise RuntimeError(f"Size mismatch for {rec['tile_id']}")
            if not geotransforms_equal(rgb_info.get("geoTransform", []), dem_info.get("geoTransform", [])):
                raise RuntimeError(f"Geotransform mismatch for {rec['tile_id']}")
            count += 1
    _log(f"Validation OK for {count} tiles")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NAIP + 3DEP dataset")
    parser.add_argument("--output-dir", required=True, help="Output root directory")
    parser.add_argument("--cache-dir", default=None, help="Cache directory (default: <output>/data/naip_3dep/cache)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cities", nargs="*", default=list(CITY_AOIS.keys()), help="Override city AOI list")
    parser.add_argument("--aoi-size-m", type=int, default=4000)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--prefer-1m-dem", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    base = output_root / "data" / "naip_3dep"
    layout = _ensure_dirs(base)
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else layout["cache"]

    if args.validate_only:
        _validate_only(layout["manifest"])
        return

    require_gdal_tools()

    aoi_list: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for name in args.cities:
        if name not in CITY_AOIS:
            raise ValueError(f"Unknown city AOI: {name}")
        aoi_list.append((name, CITY_AOIS[name]))
    aoi_list.append(_sample_mountain_aoi(args.seed, args.aoi_size_m))

    manifest: List[ManifestRecord] = []
    retries = args.max_retries
    for aoi_name, bbox in aoi_list:
        _log(f"Processing AOI {aoi_name}: {bbox}")
        naip, dem = _select_products(aoi_name, bbox, prefer_1m=bool(args.prefer_1m_dem))
        _log(f"Selected NAIP: {naip.title}")
        _log(f"Selected DEM: {dem.title}")
        if args.dry_run:
            continue
        try:
            naip_path = download_product(naip, cache_dir, logger=_log)
            dem_path = download_product(dem, cache_dir, logger=_log)
            raw_naip_dir = layout["raw_naip"] / aoi_name
            raw_dem_dir = layout["raw_dem"] / aoi_name
            raw_naip_dir.mkdir(parents=True, exist_ok=True)
            raw_dem_dir.mkdir(parents=True, exist_ok=True)
            clip_naip = raw_naip_dir / f"{aoi_name}_naip_clip.tif"
            clip_dem = raw_dem_dir / f"{aoi_name}_dem_clip.tif"
            vrt_naip = raw_naip_dir / f"{aoi_name}_naip.vrt"
            vrt_dem = raw_dem_dir / f"{aoi_name}_dem.vrt"
            _mosaic_and_clip([naip_path], bbox, vrt_naip, clip_naip)
            _mosaic_and_clip([dem_path], bbox, vrt_dem, clip_dem)

            center_lon = (bbox[0] + bbox[2]) / 2
            center_lat = (bbox[1] + bbox[3]) / 2
            epsg = derive_utm_epsg(center_lon, center_lat)
            proc_dir_rgb = layout["proc_rgb"] / aoi_name
            proc_dir_dem = layout["proc_dem"] / aoi_name
            proc_dir_rgb.mkdir(parents=True, exist_ok=True)
            proc_dir_dem.mkdir(parents=True, exist_ok=True)
            reproj_rgb = proc_dir_rgb / f"{aoi_name}_rgb_utm.tif"
            reproj_dem = proc_dir_dem / f"{aoi_name}_dem_utm.tif"
            rgb_info = _reproject_to_utm(clip_naip, epsg, reproj_rgb)
            dem_info = _align_dem_to_rgb(clip_dem, rgb_info, epsg, reproj_dem)
            if not geotransforms_equal(rgb_info.get("geoTransform", []), dem_info.get("geoTransform", [])):
                raise RuntimeError("Aligned DEM geotransform mismatch")

            _tile_pair(
                aoi_name,
                reproj_rgb,
                reproj_dem,
                proc_dir_rgb,
                proc_dir_dem,
                args.patch_size,
                args.stride,
                epsg,
                manifest,
                (naip.id, dem.id),
            )
        except Exception as exc:
            if aoi_name.startswith("rockies") and retries > 0:
                _log(f"Retrying mountainous AOI due to error: {exc}")
                retries -= 1
                new_seed = args.seed + (args.max_retries - retries)
                aoi_name, bbox = _sample_mountain_aoi(new_seed, args.aoi_size_m)
                aoi_list.append((aoi_name, bbox))
                continue
            raise

    if args.dry_run:
        _log("Dry-run complete; no data downloaded.")
        return

    _write_manifest(manifest, layout["manifest"], base)


if __name__ == "__main__":
    main()
