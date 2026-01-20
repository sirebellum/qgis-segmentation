# SPDX-License-Identifier: BSD-3-Clause
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from torch.utils.data import DataLoader

from training.config import AugConfig, DataConfig
from training.data.sharded_tif_dataset import ShardedTifDataset


def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": array.dtype,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def _build_shard(root: Path, dataset_id: str, split: str, shard_idx: int, items: list[tuple[str, int]]) -> int:
    shard_dir = root / split / dataset_id / f"shard-{shard_idx:05d}"
    entries = []
    for item_id, value in items:
        rgb = np.full((3, 4, 4), value, dtype=np.uint8)
        target = np.zeros((1, 4, 4), dtype=np.uint8)
        target[:, :2, :2] = 255
        input_path = shard_dir / "inputs" / f"{item_id}.tif"
        target_path = shard_dir / "targets" / f"{item_id}.tif"
        _write_tiff(input_path, rgb)
        _write_tiff(target_path, target)
        slic_path = shard_dir / "slic" / f"{item_id}.npz"
        slic_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(slic_path, labels=np.zeros((4, 4), dtype=np.uint16))
        entries.append(
            {
                "dataset_id": dataset_id,
                "item_id": item_id,
                "raw_split": split,
                "split": split,
                "input": str(input_path.relative_to(shard_dir)),
                "has_target": True,
                "target": str(target_path.relative_to(shard_dir)),
                "slic": str(slic_path.relative_to(shard_dir)),
            }
        )
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_path = shard_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")
    return len(entries)


def test_sharded_dataset_iterates_without_duplicates(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    total = 0
    total += _build_shard(processed, "toy", "train", 0, [("a", 1), ("b", 2)])
    total += _build_shard(processed, "toy", "train", 1, [("c", 3)])

    ds = ShardedTifDataset(
        processed_root=processed,
        dataset_id="toy",
        split="train",
        data_cfg=DataConfig(),
        aug_cfg=AugConfig(),
        with_augmentations=False,
        include_targets=True,
    )

    item_ids = []
    for sample in ds:
        item_ids.append(sample["meta"]["item_id"])
        assert sample["rgb"].shape == (3, 4, 4)
        assert sample["target"] is not None
        unique_labels = torch.unique(sample["target"]).tolist()
        assert unique_labels == [0, 1]

    assert len(item_ids) == total
    assert len(set(item_ids)) == total


def _take_first(batch):
    return batch[0]


def test_sharded_dataset_multiworker_partition(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    total = 0
    total += _build_shard(processed, "toy", "train", 0, [("a", 1), ("b", 2)])
    total += _build_shard(processed, "toy", "train", 1, [("c", 3), ("d", 4)])

    ds = ShardedTifDataset(
        processed_root=processed,
        dataset_id="toy",
        split="train",
        data_cfg=DataConfig(num_workers=2),
        aug_cfg=AugConfig(),
        with_augmentations=False,
        include_targets=False,
    )

    loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=_take_first)
    item_ids = [sample["meta"]["item_id"] for sample in loader]

    assert len(item_ids) == total
    assert len(set(item_ids)) == total


def test_cache_mode_lru_preserves_outputs(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _build_shard(processed, "toy", "train", 0, [("a", 1), ("b", 2)])

    cfg = DataConfig(cache_mode="lru", cache_max_items=1)
    ds = ShardedTifDataset(
        processed_root=processed,
        dataset_id="toy",
        split="train",
        data_cfg=cfg,
        aug_cfg=AugConfig(),
        with_augmentations=False,
        include_targets=True,
    )

    first_pass = list(ds)
    second_pass = list(ds)

    assert [s["meta"]["item_id"] for s in first_pass] == [s["meta"]["item_id"] for s in second_pass]
    assert torch.equal(first_pass[0]["rgb"], second_pass[0]["rgb"])
    assert torch.equal(first_pass[0]["target"], second_pass[0]["target"])


def test_missing_slic_requires_override(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    shard_dir = processed / "train" / "toy" / "shard-00000"
    _write_tiff(shard_dir / "inputs" / "only.tif", np.ones((3, 2, 2), dtype=np.uint8))
    shard_dir.mkdir(parents=True, exist_ok=True)
    with (shard_dir / "index.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "dataset_id": "toy",
                    "item_id": "only",
                    "raw_split": "train",
                    "split": "train",
                    "input": "inputs/only.tif",
                    "has_target": False,
                }
            )
        )

    with pytest.raises(ValueError):
        ds_required = ShardedTifDataset(
            processed_root=processed,
            dataset_id="toy",
            split="train",
            data_cfg=DataConfig(),
            aug_cfg=AugConfig(rotate_choices=(0,)),
            with_augmentations=False,
            include_targets=False,
        )
        next(iter(ds_required))

    ds_optional = ShardedTifDataset(
        processed_root=processed,
        dataset_id="toy",
        split="train",
        data_cfg=DataConfig(require_slic=False),
        aug_cfg=AugConfig(rotate_choices=(0,)),
        with_augmentations=False,
        include_targets=False,
    )
    sample = next(iter(ds_optional))
    assert "slic" in sample
    assert sample["slic"].shape == (2, 2)
