# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from training.datasets.build_shards import _build_for_header, _collect_items
from training.datasets.generate_headers import _GENERATOR_MAP
from training.datasets.header_schema import DatasetHeader, ModalitySpec, PairingSpec, ShardingSpec, SplitSpec, ValidationSpec


def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    profile = {"driver": "GTiff", "height": height, "width": width, "count": count, "dtype": array.dtype}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def _make_header(dataset_id: str, pairing: PairingSpec) -> DatasetHeader:
    return DatasetHeader(
        dataset_id=dataset_id,
        version="0.0.1",
        description=f"{dataset_id} pairing policy",
        source_root=dataset_id,
        modalities=[
            ModalitySpec(name="sat", role="input", kind="raster", pattern="{split}/images/*.tif", channels=3, dtype="uint8"),
            ModalitySpec(name="map", role="target", kind="raster", pattern="{split}/gt/*.tif", channels=1, dtype="uint8"),
        ],
        pairing=pairing,
        splits=SplitSpec(
            raw_splits=["train"],
            seed=123,
            labeled_policy="validation_only_with_metrics_subset",
            train_metric_fraction_of_labeled=0.0,
            ratios=None,
            manifest_files=None,
            preserve_raw_split=True,
        ),
        sharding=ShardingSpec(shard_size=4, layout_version=1, copy_mode="copy", force_uncompressed_tiff=True),
        validation=ValidationSpec(iou_ignore_label_leq=0, iou_average="macro_over_present_labels"),
        header_path=None,
    )


def _seed_inria_like_tree(root: Path) -> None:
    img = np.ones((3, 2, 2), dtype=np.uint8)
    mask = np.zeros((1, 2, 2), dtype=np.uint8)
    _write_tiff(root / "train" / "images" / "austin01.tif", img)
    _write_tiff(root / "train" / "images" / "austin02.tif", img * 2)
    _write_tiff(root / "train" / "gt" / "austin01.tif", mask)
    _write_tiff(root / "train" / "gt" / "austin24.tif", mask)


def test_collect_items_drops_target_only_by_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    extracted = tmp_path / "extracted"
    dataset_root = extracted / "pairing"
    _seed_inria_like_tree(dataset_root)

    pairing = PairingSpec(strategy="by_stem", input_modality="sat", target_modality="map")
    header = _make_header("pairing", pairing)

    items = _collect_items(header, dataset_root)
    out = capsys.readouterr().out

    item_ids = [item.item_id for item in items]
    assert item_ids == ["austin01", "austin02"]
    assert "austin24" in out  # dropped target-only tile should be logged

    has_target = {item.item_id: bool(item.target_path) for item in items}
    assert has_target == {"austin01": True, "austin02": False}


def test_collect_items_errors_in_strict_mode(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    dataset_root = extracted / "pairing_strict"
    _seed_inria_like_tree(dataset_root)

    pairing = PairingSpec(
        strategy="by_stem",
        input_modality="sat",
        target_modality="map",
        on_missing_input="error",
    )
    header = _make_header("pairing_strict", pairing)

    with pytest.raises(ValueError):
        _collect_items(header, dataset_root)


def test_inria_header_generator_sets_pairing_policy(tmp_path: Path) -> None:
    source_root = tmp_path / "inria"
    train_img_dir = source_root / "train" / "images"
    train_gt_dir = source_root / "train" / "gt"
    test_img_dir = source_root / "test" / "images"

    img = np.ones((3, 2, 2), dtype=np.uint8)
    mask = np.zeros((1, 2, 2), dtype=np.uint8)
    _write_tiff(train_img_dir / "tile_a.tif", img)
    _write_tiff(train_gt_dir / "tile_a.tif", mask)
    _write_tiff(train_gt_dir / "orphan.tif", mask)
    _write_tiff(test_img_dir / "tile_b.tif", img * 2)

    generator = _GENERATOR_MAP["inria"]
    header, stats = generator(source_root)

    assert header.pairing.on_missing_input == "drop_item"
    assert header.pairing.on_missing_target == "allow"

    pairing_stats = stats.get("pairing", {})
    assert pairing_stats.get("train", {}).get("target_only") == ["orphan"]


def test_build_shards_skips_target_only_items(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    processed = tmp_path / "processed"
    dataset_root = extracted / "pairing_shard"
    _seed_inria_like_tree(dataset_root)

    pairing = PairingSpec(strategy="by_stem", input_modality="sat", target_modality="map")
    header = _make_header("pairing_shard", pairing)

    summary = _build_for_header(
        header,
        extracted_root=extracted,
        output_root=processed,
        shard_size=None,
        seed_override=1,
        max_items=None,
        overwrite=True,
        dry_run=False,
    )

    assert summary["train"]["items"] == 2

    shard_root = processed / "train" / "pairing_shard"
    index_path = next(shard_root.glob("**/index.jsonl"))
    entries = [json.loads(line) for line in index_path.read_text().splitlines()]

    ids = {entry["item_id"] for entry in entries}
    assert ids == {"austin01", "austin02"}
    missing_targets = {entry["item_id"] for entry in entries if not entry.get("has_target", False)}
    assert missing_targets == {"austin02"}
