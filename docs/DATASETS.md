<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# NAIP + 3DEP Dataset (Training Only)

- Source layers: NAIP RGB imagery + USGS 3DEP DEM (prefers 1 m, falls back to 1/3 arc-second then 1 arc-second).
- Script: `scripts/data/prepare_naip_3dep_dataset.py` downloads four AOIs (3 cities + 1 seeded mountain), co-registers RGB/DEM to UTM, tiles 512x512 (stride 128), and writes `manifest.jsonl`.
- Output layout (under `--output-dir`):
  - `data/naip_3dep/raw/naip/<aoi>/...`, `data/naip_3dep/raw/dem/<aoi>/...`
  - `data/naip_3dep/processed/rgb/<aoi>/<tile>.tif`, `processed/dem/<aoi>/<tile>.tif`
  - `data/naip_3dep/processed/manifest.jsonl`
  - `data/naip_3dep/logs/`, `data/naip_3dep/cache/`
- Manifest schema (jsonl, one object per tile):
  - `aoi_name`, `tile_id`, `rgb_path`, `dem_path` (relative to output root)
  - `epsg`, `pixel_size`, `width`, `height`, `bounds`, `geotransform`
  - `source_naip`, `source_dem`, `dem_tier` (optional)

## Quickstart
1) Dry-run (query only):
```
python scripts/data/prepare_naip_3dep_dataset.py --output-dir /tmp/naip3dep --dry-run --seed 123
```
2) Download + tile:
```
python scripts/data/prepare_naip_3dep_dataset.py --output-dir /tmp/naip3dep --seed 123 --aoi-size-m 4000 --patch-size 512 --stride 128
```
3) Validate alignment + manifest:
```
python scripts/data/prepare_naip_3dep_dataset.py --output-dir /tmp/naip3dep --validate-only
```

## Notes
- Requires GDAL CLI tools (`gdalinfo`, `gdalwarp`, `gdal_translate`, `gdalbuildvrt`); install via `brew install gdal` or `apt-get install gdal-bin`.
- Endpoint logic isolated in `_usgs_tnm_provider.py`; swap endpoints there if TNM changes.
- User is responsible for data licensing/compliance for NAIP and 3DEP assets.
- No QGIS runtime inference changes are introduced by this dataset pipeline.
