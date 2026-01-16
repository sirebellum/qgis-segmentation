<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Dataset CODE_DESCRIPTION

- scripts/data/_gdal_utils.py: GDAL CLI wrappers (require_gdal_tools, run_cmd, gdalinfo_json), grid helpers (GridSpec, warp_to_grid, bounds/pixel size/UTM epsg utilities), and vector subset helper for NAIP footprint clipping.
- scripts/data/_naip_aws_provider.py: NAIP-on-AWS index helper with requester-pays VRT build, GeoJSON-first querying (no GDAL needed), bbox overlap filter, and embedded stub index for dry-run.
- scripts/data/_usgs_3dep_provider.py: TNM Access client for DEM tier ladder (1 m → 1/3" → 1"), product selection, and download-to-cache helper.
- scripts/data/prepare_naip_aws_3dep_dataset.py: AWS NAIP + 3DEP prep CLI (warp to shared grid, tile, manifest); `--dry-run` now skips GDAL checks, uses stub index/DEM, and logs selections only.
- scripts/data/prepare_naip_3dep_dataset.py: legacy TNM NAIP + 3DEP prep/tiler (GDAL + network required).
- tests/test_prepare_naip_aws_3dep_dry_run.py: offline dry-run coverage with embedded stub index; uses fixture tests/fixtures/naip_index_min.geojson.
