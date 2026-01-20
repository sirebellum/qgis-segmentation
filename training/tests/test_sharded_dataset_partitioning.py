# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from training.config import AugConfig, DataConfig
from training.data.sharded_tif_dataset import ShardedTifDataset


def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    profile = {"driver": "GTiff", "height": height, "width": width, "count": count, "dtype": array.dtype}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def _write_slic(path: Path, size: tuple[int, int] = (4, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, labels=np.zeros(size, dtype=np.uint16))


def _build_index(shard_dir: Path, entries: list[dict]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_path = shard_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")


def test_include_targets_filters_missing_entries(tmp_path: Path) -> None:
    shard = tmp_path / "processed" / "train" / "toy" / "shard-00000"
    rgb = np.ones((3, 4, 4), dtype=np.uint8)
    target = np.ones((1, 4, 4), dtype=np.uint8)
    _write_tiff(shard / "inputs" / "with_target.tif", rgb)
    _write_tiff(shard / "targets" / "with_target.tif", target)
    _write_tiff(shard / "inputs" / "no_target.tif", rgb * 2)
    _write_slic(shard / "slic" / "with_target.npz", size=(4, 4))
    _write_slic(shard / "slic" / "no_target.npz", size=(4, 4))

    entries = [
        {
            "dataset_id": "toy",
            "item_id": "with_target",
            "raw_split": "train",
            "split": "train",
            "input": "inputs/with_target.tif",
            "has_target": True,
            "target": "targets/with_target.tif",
            "slic": "slic/with_target.npz",
        },
        {
            "dataset_id": "toy",
            "item_id": "no_target",
            "raw_split": "train",
            "split": "train",
            "input": "inputs/no_target.tif",
            "has_target": False,
            "slic": "slic/no_target.npz",
        },
    ]
    _build_index(shard, entries)

    cfg = DataConfig(max_samples=None)
    ds_all = ShardedTifDataset(
        processed_root=tmp_path / "processed",
        dataset_id="toy",
        split="train",
        data_cfg=cfg,
        aug_cfg=AugConfig(rotate_choices=(0,)),
        with_augmentations=False,
        include_targets=False,
    )
    all_items = [sample["meta"]["item_id"] for sample in ds_all]
    assert set(all_items) == {"with_target", "no_target"}

    ds_targets_only = ShardedTifDataset(
        processed_root=tmp_path / "processed",
        dataset_id="toy",
        split="train",
        data_cfg=cfg,
        aug_cfg=AugConfig(rotate_choices=(0,)),
        with_augmentations=False,
        include_targets=True,
    )
    filtered = list(ds_targets_only)
    assert [sample["meta"]["item_id"] for sample in filtered] == ["with_target"]
    assert torch.unique(filtered[0]["target"]).tolist() == [1]


def test_max_samples_limits_iteration(tmp_path: Path) -> None:
    shard = tmp_path / "processed" / "train" / "toy" / "shard-00000"
    entries = []
    for idx in range(3):
        _write_tiff(shard / "inputs" / f"item_{idx}.tif", np.full((3, 2, 2), idx, dtype=np.uint8))
        _write_slic(shard / "slic" / f"item_{idx}.npz", size=(2, 2))
        entries.append(
            {
                "dataset_id": "toy",
                "item_id": f"item_{idx}",
                "raw_split": "train",
                "split": "train",
                "input": f"inputs/item_{idx}.tif",
                "has_target": False,
                "slic": f"slic/item_{idx}.npz",
            }
        )
    _build_index(shard, entries)

    cfg = DataConfig(max_samples=2)
    ds = ShardedTifDataset(
        processed_root=tmp_path / "processed",
        dataset_id="toy",
        split="train",
        data_cfg=cfg,
        aug_cfg=AugConfig(rotate_choices=(0,)),
        with_augmentations=False,
        include_targets=False,
    )

    assert len(list(ds)) == 2
