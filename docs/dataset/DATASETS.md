<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# NAIP + 3DEP Datasets (Training Only)

## NAIP on AWS + 3DEP (COG, Requester Pays)
- Sources: NAIP AWS COGs (bucket `naip-visualization`, alt `naip-analytic`/`naip-source`, Requester Pays, us-west-2) + USGS 3DEP DEM (TNM Access API https://tnmaccess.usgs.gov/api/v1/products).
- Index: cached footprint `naip_footprint.gpkg` under `<cache>/index/`; auto-download from `https://naip-visualization.s3.amazonaws.com/index/naip_footprint.gpkg` unless `--naip-index-path` or `--naip-index-source` overrides. `AWS_REQUEST_PAYER=requester` is set for GDAL vsis3/HTTPS.
- Script: `scripts/data/prepare_naip_aws_3dep_dataset.py` selects 3 city AOIs + 1 seeded mountain patch, builds a VRT over NAIP COGs, warps RGB to a single grid (-tap, target CRS auto UTM or `--target-crs`, pixel size `--target-gsd` but never finer than DEM native), warps DEM to that grid, tiles 512x512 (stride 128), drops tiles over `--max-nodata-frac`, and writes `manifest.jsonl` with DEM tier/native/resampled flags.
- DEM tier ladder: prefers 1 m (`3DEP 1 meter DEM`, `USGS 1 meter x 1 meter Resolution DEM`), falls back to 1/3 arc-second (~10 m), then 1 arc-second (~30 m). Selection recorded in manifest.
- Layout (under `--output-dir` → `data/naip_aws_3dep/`): `raw/naip/`, `raw/dem/`, `processed/rgb/`, `processed/dem/`, `processed/manifest.jsonl`, `logs/`, `cache/` (index + downloads).
- Manifest fields per tile: `aoi_name`, `tile_id`, `rgb_path`, `dem_path`, `epsg`, `pixel_size`, `bounds`, `geotransform`, `nodata_fraction`, `source_naip_urls`, `source_naip_year/state`, `source_dem_id`, `dem_tier`, `dem_native_gsd`, `dem_target_gsd`, `dem_resampled`.
- Dry-run: skips GDAL/tool checks and uses an embedded GeoJSON stub index plus stub DEM selection; no downloads are attempted. Full runs still require GDAL CLIs and network access.

Quickstart:
```
python scripts/data/prepare_naip_aws_3dep_dataset.py --output-dir /tmp/naipaws3dep --dry-run --seed 123
python scripts/data/prepare_naip_aws_3dep_dataset.py --output-dir /tmp/naipaws3dep --seed 123 --aoi-size-m 4000 --patch-size 512 --stride 128
python scripts/data/prepare_naip_aws_3dep_dataset.py --output-dir /tmp/naipaws3dep --validate --sample-tiles 10
```

Minimal real download (no AWS/TNM creds):
```
python scripts/data/prepare_naip_aws_3dep_dataset.py --output-dir /tmp/naipaws3dep --sample-data --patch-size 32 --stride 32 --target-gsd 30 --use-https --seed 7
```
- Uses public GitHub sample rasters (`rgbsmall.tif`, `byte.tif`) via HTTPS; still runs the real warp/tile/manifest pipeline with tiny artifacts under `data/naip_aws_3dep/`.
- For full AWS runs, provide AWS credentials (Requester Pays) or `--naip-index-path`; TNM access remains required for DEM discovery unless `--dem-url` is supplied.

## TNM NAIP + 3DEP (legacy)
- Script: `scripts/data/prepare_naip_3dep_dataset.py` (TNM NAIP download) remains for backward compatibility. Manifest schema is a subset of the AWS version and still works with the loader.

## General Notes
- Requires GDAL CLI tools (`gdalinfo`, `gdalwarp`, `gdal_translate`, `gdalbuildvrt`, `ogr2ogr`).
- User is responsible for licensing/compliance (NAIP is public domain; AWS Requester Pays may incur charges; 3DEP products are free).
- Training-only; QGIS runtime inference remains unchanged.
