# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from training.config import AugConfig, DataConfig
from training.data.naip_3dep_dataset import Naip3DepDataset


def _write_geo_tiff(path: Path, array: np.ndarray, transform, crs: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = array.shape[0]
    height, width = array.shape[1:]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array)


def test_manifest_loader_reads_aligned_tiles(tmp_path):
    rgb = (np.random.rand(3, 32, 32) * 255).astype(np.uint8)
    dem = np.random.rand(1, 32, 32).astype(np.float32)
    transform = from_origin(0, 10, 1, 1)
    crs = "EPSG:32615"

    rgb_path = tmp_path / "data" / "naip_3dep" / "processed" / "rgb" / "aoi" / "tile.tif"
    dem_path = tmp_path / "data" / "naip_3dep" / "processed" / "dem" / "aoi" / "tile.tif"
    _write_geo_tiff(rgb_path, rgb, transform, crs)
    _write_geo_tiff(dem_path, dem, transform, crs)

    manifest_path = tmp_path / "data" / "naip_3dep" / "processed" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "aoi_name": "aoi",
        "tile_id": "tile",
        "rgb_path": str(rgb_path.relative_to(tmp_path)),
        "dem_path": str(dem_path.relative_to(tmp_path)),
        "epsg": 32615,
        "pixel_size": (1.0, -1.0),
        "width": 32,
        "height": 32,
        "bounds": (0.0, -22.0, 32.0, 10.0),
        "geotransform": (0.0, 1.0, 0.0, 10.0, 0.0, -1.0),
        "source_naip": "naip_test",
        "source_dem": "dem_test",
    }
    with manifest_path.open("w") as f:
        f.write(json.dumps(record) + "\n")

    cfg = DataConfig(patch_size=32, stride=32, manifest_path=str(manifest_path), standardize_elevation=True)
    aug_cfg = AugConfig()
    ds = Naip3DepDataset(manifest_path, cfg, aug_cfg, validate=True)
    sample = ds[0]

    assert sample["view1"]["rgb"].shape[-2:] == (32, 32)
    assert sample["view2"]["rgb"].shape[-2:] == (32, 32)
    assert sample["view1"].get("elev") is None or sample["view1"]["elev"].shape[-2:] == (32, 32)
    grid = sample["warp_grid"]
    assert grid.shape[1:3] == (32, 32)
    assert grid.shape[-1] == 2
