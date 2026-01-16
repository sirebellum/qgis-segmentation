# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import json
from pathlib import Path

from training.data.naip_3dep_dataset import load_manifest


def test_load_manifest_parses_extended_fields(tmp_path):
    root = tmp_path / "dataset"
    tiles = root / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    rgb = tiles / "rgb.tif"
    dem = tiles / "dem.tif"
    rgb.write_bytes(b"")
    dem.write_bytes(b"")

    record = {
        "aoi_name": "test-aoi",
        "tile_id": "tile-0",
        "rgb_path": "tiles/rgb.tif",
        "dem_path": "tiles/dem.tif",
        "epsg": 32613,
        "pixel_size": (1.0, -1.0),
        "width": 4,
        "height": 4,
        "bounds": (0.0, 0.0, 4.0, 4.0),
        "geotransform": (0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
        "source_naip": "aws-footprint",
        "source_dem": "3dep-1m",
        "nodata_fraction": 0.1,
        "source_naip_urls": ["https://example.com/a.tif"],
        "source_naip_year": "2021",
        "source_naip_state": "CO",
        "dem_native_gsd": 1.0,
        "dem_resampled": True,
        "dem_target_gsd": 2.0,
    }

    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    entries = load_manifest(manifest, anchor=root)
    assert len(entries) == 1
    entry = entries[0]

    assert entry.tile_id == record["tile_id"]
    assert entry.rgb_path == rgb.resolve()
    assert entry.dem_path == dem.resolve()
    assert entry.pixel_size == tuple(record["pixel_size"])
    assert entry.bounds == tuple(record["bounds"])
    assert entry.geotransform == tuple(record["geotransform"])
    assert entry.nodata_fraction == record["nodata_fraction"]
    assert entry.source_naip_urls == record["source_naip_urls"]
    assert entry.dem_resampled is True
    assert entry.dem_target_gsd == record["dem_target_gsd"]
    assert entry.dem_native_gsd == record["dem_native_gsd"]
