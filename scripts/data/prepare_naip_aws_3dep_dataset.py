# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Prepare NAIP (AWS COG) + USGS 3DEP DEM tiles for training.

Differences from the TNM-based Phase 5 script:
- NAIP source is the AWS Requester Pays COG distribution (naip-visualization by default).
- NAIP coverage is discovered via a cached index (GeoPackage/GeoJSON); users can
  supply a local index path or force refresh from the official footprint.
- DEM discovery still uses TNM Access (https://tnmaccess.usgs.gov/api/v1/) with a
  tier ladder (1m -> 1/3" -> 1").
- Alignment is strict: derive a single grid from the RGB warp (-tap, target CRS
  + pixel size), then warp DEM to that grid with matching -tr/-te/-ts.
- Tiles are 512x512 with stride 128 by default; tiles exceeding --max-nodata-frac
  are skipped. Manifest records dem/native/resampled metadata.

Notes:
- Requester Pays buckets require AWS_REQUEST_PAYER=requester (set automatically
  for GDAL calls here). If credentials are needed, export AWS_ACCESS_KEY_ID and
  AWS_SECRET_ACCESS_KEY before running.
- This script is training-only; it does not touch QGIS runtime inference paths.
"""
from __future__ import annotations

import argparse
import json
import os
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Allow stand-alone execution
if __package__ in {None, ""}:
    import sys

    _here = Path(__file__).resolve()
    _root = _here.parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

try:
    from ._gdal_utils import (
        GridSpec,
        bounds_from_info,
        derive_utm_epsg,
        gdalinfo_json,
        geotransform_from_info,
        geotransforms_equal,
        pixel_size_from_info,
        require_gdal_tools,
        run_cmd,
        warp_to_grid,
    )
    from ._naip_aws_provider import NaipAwsConfig, NaipIndex, NaipTileRef, build_naip_vrt
    from ._usgs_3dep_provider import DemSelection, download_product, query_products, select_best_dem
except ImportError:  # pragma: no cover - fallback when executed directly
    from scripts.data._gdal_utils import (
        GridSpec,
        bounds_from_info,
        derive_utm_epsg,
        gdalinfo_json,
        geotransform_from_info,
        geotransforms_equal,
        pixel_size_from_info,
        require_gdal_tools,
        run_cmd,
        warp_to_grid,
    )
    from scripts.data._naip_aws_provider import NaipAwsConfig, NaipIndex, NaipTileRef, build_naip_vrt
    from scripts.data._usgs_3dep_provider import DemSelection, download_product, query_products, select_best_dem

try:  # Optional dependency for nodata stats
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

# Fixed AOIs (lon/lat EPSG:4326)
CITY_AOIS: Dict[str, Tuple[float, float, float, float]] = {
    "chicago_il": (-87.72, 41.84, -87.60, 41.92),
    "phoenix_az": (-112.15, 33.39, -111.95, 33.55),
    "atlanta_ga": (-84.45, 33.70, -84.34, 33.82),
}
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
    nodata_fraction: float
    source_naip_urls: List[str]
    source_naip_year: Optional[str]
    source_naip_state: Optional[str]
    source_dem_id: Optional[str]
    dem_tier: Optional[str]
    dem_native_gsd: Optional[float]
    dem_resampled: bool
    dem_target_gsd: float


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


def _target_epsg(bbox_wgs84: Sequence[float], override: Optional[int]) -> int:
    if override:
        return int(override)
    lon = (bbox_wgs84[0] + bbox_wgs84[2]) / 2
    lat = (bbox_wgs84[1] + bbox_wgs84[3]) / 2
    return derive_utm_epsg(lon, lat)


def _grid_from_info(info: dict) -> GridSpec:
    px = pixel_size_from_info(info)
    bounds = bounds_from_info(info)
    width, height = info.get("size", [0, 0])
    coord = info.get("coordinateSystem") or {}
    epsg = None
    if isinstance(coord, dict):
        epsg = coord.get("epsg") or coord.get("authorityCode")
        if epsg is None and "wkt" in coord:
            text = str(coord["wkt"])
            if "EPSG" in text:
                try:
                    epsg = int(text.split("EPSG")[-1].split(")")[0].split(":")[-1])
                except Exception:
                    epsg = None
    if epsg is None:
        epsg = 0
    return GridSpec(epsg=int(epsg), pixel_size=px, bounds=bounds, size=(width, height))


def _nodata_fraction(tile_path: Path) -> float:
    if rasterio is None:
        return 0.0
    with rasterio.open(tile_path) as src:
        data = src.read(masked=True)
    if data.size == 0:
        return 1.0
    if hasattr(data, "mask"):
        mask = data.mask
        return float(mask.sum()) / float(mask.size)
    return 0.0


def _write_manifest(manifest: List[ManifestRecord], manifest_path: Path, root: Path) -> None:
    anchor = root.parent.parent if root.parent.parent.exists() else root
    with manifest_path.open("w") as f:
        for rec in manifest:
            record = asdict(rec)
            record["rgb_path"] = str(Path(record["rgb_path"]).resolve().relative_to(anchor))
            record["dem_path"] = str(Path(record["dem_path"]).resolve().relative_to(anchor))
            f.write(json.dumps(record) + "\n")
    _log(f"Wrote manifest with {len(manifest)} entries: {manifest_path}")


def _validate_tiles(manifest_path: Path, sample: int) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    anchor = manifest_path.parent.parent.parent if manifest_path.parent.parent.parent.exists() else manifest_path.parent
    count = 0
    with manifest_path.open() as f:
        for line in f:
            rec = json.loads(line)
            rgb = Path(anchor / rec["rgb_path"]).resolve()
            dem = Path(anchor / rec["dem_path"]).resolve()
            rgb_info = gdalinfo_json(rgb)
            dem_info = gdalinfo_json(dem)
            if rgb_info.get("size") != dem_info.get("size"):
                raise RuntimeError(f"Size mismatch for {rec['tile_id']}")
            if not geotransforms_equal(rgb_info.get("geoTransform", []), dem_info.get("geoTransform", [])):
                raise RuntimeError(f"Geotransform mismatch for {rec['tile_id']}")
            count += 1
            if sample and count >= sample:
                break
    _log(f"Validation OK for {count} tile(s)")


def _select_naip_tiles(index: NaipIndex, bbox: Sequence[float]) -> List[NaipTileRef]:
    tiles = index.query(bbox)
    if not tiles:
        raise RuntimeError("No NAIP tiles intersect AOI")
    return tiles


def _select_dem(bbox: Sequence[float], prefer_1m: bool, logger) -> DemSelection:
    from ._usgs_3dep_provider import DEM_TIERS

    all_products = []
    for tier in DEM_TIERS:
        for ds in tier["datasets"]:  # type: ignore[index]
            all_products.extend(query_products(ds, bbox, logger=logger))
    selection = select_best_dem(all_products, prefer_1m=prefer_1m)
    if not selection:
        raise RuntimeError("No DEM products found for AOI")
    return selection


def _rgb_target_pixel_size(requested: float, dem_native: Optional[float]) -> float:
    if dem_native is None:
        return requested
    return max(requested, dem_native)


def _warp_naip(vrt: Path, out_path: Path, bbox_wgs84: Sequence[float], grid_px: float, epsg: int, env: Optional[dict], logger) -> dict:
    args = [
        "gdalwarp",
        str(vrt),
        str(out_path),
        "-t_srs",
        f"EPSG:{epsg}",
        "-te_srs",
        "EPSG:4326",
        "-te",
        str(bbox_wgs84[0]),
        str(bbox_wgs84[1]),
        str(bbox_wgs84[2]),
        str(bbox_wgs84[3]),
        "-tr",
        str(grid_px),
        str(grid_px),
        "-tap",
        "-r",
        "bilinear",
        "-overwrite",
    ]
    run_cmd(args, logger=logger, env=env)
    return gdalinfo_json(out_path)


def _tile_pair(aoi: str, rgb: Path, dem: Path, out_rgb_dir: Path, out_dem_dir: Path, patch: int, stride: int, epsg: int, manifest: List[ManifestRecord], source_naip_tiles: List[NaipTileRef], dem_selection: DemSelection, grid_px: float, max_nodata_frac: float) -> None:
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
                "COMPRESS=JPEG",
                "-co",
                "TILED=YES",
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
                "COMPRESS=DEFLATE",
                "-co",
                "TILED=YES",
            ], logger=_log)
            nodata_frac = _nodata_fraction(rgb_tile)
            if nodata_frac > max_nodata_frac:
                rgb_tile.unlink(missing_ok=True)
                dem_tile.unlink(missing_ok=True)
            else:
                tile_minx = base_gt[0] + x * base_gt[1]
                tile_maxy = base_gt[3] + y * base_gt[5]
                tile_maxx = tile_minx + patch * base_gt[1]
                tile_miny = tile_maxy + patch * base_gt[5]
                urls = [t.url for t in source_naip_tiles]
                px_out = (abs(px_size[0]), abs(px_size[1])) if px_size else px_size
                first_tile = source_naip_tiles[0] if source_naip_tiles else None
                manifest.append(
                    ManifestRecord(
                        aoi_name=aoi,
                        tile_id=tile_id,
                        rgb_path=str(rgb_tile),
                        dem_path=str(dem_tile),
                        epsg=epsg,
                        pixel_size=px_out,
                        width=patch,
                        height=patch,
                        bounds=(tile_minx, tile_miny, tile_maxx, tile_maxy),
                        geotransform=base_gt,
                        nodata_fraction=nodata_frac,
                        source_naip_urls=urls,
                        source_naip_year=first_tile.year if first_tile else None,
                        source_naip_state=first_tile.state if first_tile else None,
                        source_dem_id=dem_selection.product.id if dem_selection else None,
                        dem_tier=dem_selection.product.tier_label if dem_selection else None,
                        dem_native_gsd=dem_selection.native_gsd if dem_selection else None,
                        dem_resampled=bool(dem_selection.native_gsd and abs(dem_selection.native_gsd - grid_px) > 1e-3),
                        dem_target_gsd=abs(grid_px),
                    )
                )
            y += stride
        x += stride
    _log(f"Tiled {aoi}: {len(manifest)} total tiles so far")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NAIP AWS + 3DEP dataset")
    parser.add_argument("--output-dir", required=True, help="Output root directory")
    parser.add_argument("--cache-dir", default=None, help="Cache directory (default: <output>/data/naip_aws_3dep/cache)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cities", nargs="*", default=list(CITY_AOIS.keys()), help="Override city AOI list")
    parser.add_argument("--aoi-size-m", type=int, default=4000)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--prefer-1m-dem", type=int, default=1)
    parser.add_argument("--target-crs", type=int, default=None, help="Optional EPSG override for all AOIs")
    parser.add_argument("--target-gsd", type=float, default=1.0, help="Target pixel size (meters); may be increased to match DEM native")
    parser.add_argument("--naip-index-path", type=str, default=None, help="Path to NAIP index (GeoPackage/GeoJSON)")
    parser.add_argument("--naip-index-source", type=str, default=None, help="Override NAIP index URL (default: official footprint)")
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--sample-tiles", type=int, default=8)
    parser.add_argument("--max-nodata-frac", type=float, default=0.25)
    parser.add_argument("--use-https", action="store_true", help="Use HTTPS instead of vsis3 for NAIP COGs")
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    base = output_root / "data" / "naip_aws_3dep"
    layout = _ensure_dirs(base)
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else layout["cache"]

    if args.validate_only:
        _validate_tiles(layout["manifest"], args.sample_tiles)
        return

    require_gdal_tools()

    naip_cfg = NaipAwsConfig(index_url=args.naip_index_source)
    naip_index = NaipIndex(cache_dir / "index", naip_cfg, index_path=Path(args.naip_index_path) if args.naip_index_path else None, refresh=args.refresh_index, logger=_log)

    aoi_list: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for name in args.cities:
        if name not in CITY_AOIS:
            raise ValueError(f"Unknown city AOI: {name}")
        aoi_list.append((name, CITY_AOIS[name]))
    aoi_list.append(_sample_mountain_aoi(args.seed, args.aoi_size_m))

    manifest: List[ManifestRecord] = []
    retries = args.max_retries

    env_req = None
    if naip_cfg.requester_pays:
        env_req = dict(os.environ)
        env_req["AWS_REQUEST_PAYER"] = "requester"
        env_req.setdefault("AWS_REGION", naip_cfg.region)

    for aoi_name, bbox in aoi_list:
        _log(f"Processing AOI {aoi_name}: {bbox}")
        try:
            naip_tiles = _select_naip_tiles(naip_index, bbox)
            naip_urls = [tile.gdal_source(use_vsis3=not args.use_https) for tile in naip_tiles]
            dem_selection = _select_dem(bbox, prefer_1m=bool(args.prefer_1m_dem), logger=_log)
            if args.dry_run:
                _log(f"Dry-run: selected {len(naip_tiles)} NAIP tiles and DEM {dem_selection.product.title}")
                continue
            raw_naip_dir = layout["raw_naip"] / aoi_name
            raw_dem_dir = layout["raw_dem"] / aoi_name
            raw_naip_dir.mkdir(parents=True, exist_ok=True)
            raw_dem_dir.mkdir(parents=True, exist_ok=True)
            vrt_naip = raw_naip_dir / f"{aoi_name}_naip.vrt"
            build_naip_vrt(naip_urls, vrt_naip, requester_pays=naip_cfg.requester_pays, logger=_log)

            dem_products = dem_selection.product
            dem_path = download_product(dem_products, cache_dir, logger=_log)
            dem_vrt = raw_dem_dir / f"{aoi_name}_dem.vrt"
            run_cmd(["gdalbuildvrt", str(dem_vrt), str(dem_path)], logger=_log)

            target_epsg = _target_epsg(bbox, args.target_crs)
            final_px = _rgb_target_pixel_size(args.target_gsd, dem_selection.native_gsd)
            proc_dir_rgb = layout["proc_rgb"] / aoi_name
            proc_dir_dem = layout["proc_dem"] / aoi_name
            proc_dir_rgb.mkdir(parents=True, exist_ok=True)
            proc_dir_dem.mkdir(parents=True, exist_ok=True)
            reproj_rgb = proc_dir_rgb / f"{aoi_name}_rgb.tif"
            reproj_dem = proc_dir_dem / f"{aoi_name}_dem.tif"
            rgb_info = _warp_naip(vrt_naip, reproj_rgb, bbox, final_px, target_epsg, env_req, _log)
            grid = _grid_from_info(rgb_info)
            warp_to_grid(dem_vrt, reproj_dem, grid, resampling="bilinear", dst_nodata=-9999, env=env_req, logger=_log)
            dem_info = gdalinfo_json(reproj_dem)
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
                target_epsg,
                manifest,
                naip_tiles,
                dem_selection,
                grid.pixel_size[0],
                args.max_nodata_frac,
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
    if args.validate:
        _validate_tiles(layout["manifest"], args.sample_tiles)


if __name__ == "__main__":
    main()
