# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
import rasterio
import torch

from training.config import AugConfig, DataConfig
from training.data.geo_patch_dataset import GeoPatchViewsDataset, GeoTiffPatchDataset


def _write_rgb_tif(path, array):
    height, width = array.shape[1], array.shape[2]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=array.dtype,
    ) as dst:
        for idx in range(3):
            dst.write(array[idx], idx + 1)


def _write_target_tif(path, array):
    height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=array.dtype,
    ) as dst:
        dst.write(array, 1)


def test_geo_patch_requires_raster_path(tmp_path):
    with pytest.raises(ValueError):
        GeoTiffPatchDataset([], data_cfg=DataConfig(patch_size=8))


def test_geo_patch_rejects_mismatched_targets(tmp_path):
    rgb_path = tmp_path / "rgb.tif"
    rgb = np.zeros((3, 8, 8), dtype=np.uint8)
    _write_rgb_tif(rgb_path, rgb)

    with pytest.raises(ValueError):
        GeoTiffPatchDataset([rgb_path], targets=[rgb_path, rgb_path], data_cfg=DataConfig(patch_size=8))


def test_geo_patch_loads_rgb_and_target(tmp_path):
    rgb_path = tmp_path / "rgb.tif"
    target_path = tmp_path / "target.tif"
    rgb = np.stack([np.arange(64, dtype=np.uint8).reshape(8, 8)] * 3, axis=0)
    target = np.full((8, 8), 7, dtype=np.uint8)
    _write_rgb_tif(rgb_path, rgb)
    _write_target_tif(target_path, target)

    dataset = GeoTiffPatchDataset([rgb_path], targets=[target_path], data_cfg=DataConfig(patch_size=8), with_targets=True)
    sample = dataset[0]

    assert sample.rgb.shape == (3, 8, 8)
    assert torch.is_tensor(sample.rgb)
    assert sample.rgb.dtype == torch.float32
    assert sample.rgb.max() <= 1.0
    assert sample.target is not None
    assert sample.target.shape == (8, 8)
    assert sample.target.dtype == torch.int64
    assert sample.meta["window"][2:] == (8, 8)
    assert str(rgb_path) in sample.meta["raster"]


def test_views_dataset_rejects_non_right_angle_rotations(tmp_path):
    rgb_path = tmp_path / "rgb.tif"
    rgb = np.ones((3, 8, 8), dtype=np.uint8)
    _write_rgb_tif(rgb_path, rgb)

    base = GeoTiffPatchDataset([rgb_path], data_cfg=DataConfig(patch_size=8))
    views = GeoPatchViewsDataset(base, aug_cfg=AugConfig(rotate_choices=(45,)))

    with pytest.raises(ValueError):
        _ = views[0]


def test_geo_patch_supports_multiple_patch_sizes(tmp_path):
    rgb_path = tmp_path / "rgb_large.tif"
    rgb = np.ones((3, 32, 32), dtype=np.uint8)
    _write_rgb_tif(rgb_path, rgb)

    for patch in (8, 16):
        dataset = GeoTiffPatchDataset([rgb_path], data_cfg=DataConfig(patch_size=patch))
        sample = dataset[0]
        assert sample.rgb.shape[-2:] == (patch, patch)
        assert sample.meta["patch_size"] == patch
