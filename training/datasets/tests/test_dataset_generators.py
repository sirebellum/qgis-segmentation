# SPDX-License-Identifier: BSD-3-Clause
import json
from pathlib import Path

import numpy as np
import rasterio

from training.datasets.build_shards import _build_for_header
from training.datasets.header_schema import (
    DatasetHeader,
    ModalitySpec,
    PairingSpec,
    ShardingSpec,
    SplitSpec,
    ValidationSpec,
    _parse_modality,
    _parse_pairing,
    _parse_splits,
    _parse_sharding,
    _parse_validation,
)


def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": array.dtype,
        "crs": "EPSG:4326",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def _read_first_index(shard_root: Path) -> tuple[dict, Path]:
    index_path = next(shard_root.glob("**/index.jsonl"))
    line = index_path.read_text().splitlines()[0]
    return json.loads(line), index_path.parent


def test_header_parsing_minimum_fields(tmp_path: Path) -> None:
    raw_header = {
        "dataset_id": "demo_ds",
        "version": "0.0.1",
        "description": "header parse test",
        "source_root": "demo",
        "modalities": [
            {"name": "sat", "role": "input", "kind": "raster", "pattern": "{split}/sat/*.tiff", "channels": 3, "dtype": "uint8"},
            {"name": "map", "role": "target", "kind": "raster", "pattern": "{split}/map/*.tif", "channels": 1, "dtype": "uint8"},
        ],
        "pairing": {"strategy": "by_stem", "input_modality": "sat", "target_modality": "map"},
        "splits": {"raw": ["training"], "ratios": {"train": 0.7, "val": 0.2, "test": 0.1}},
        "sharding": {"shard_size": 8},
        "validation": {},
    }

    modalities = [_parse_modality(entry) for entry in raw_header["modalities"]]
    pairing = _parse_pairing(raw_header["pairing"], modalities)
    splits = _parse_splits(raw_header["splits"])
    sharding = _parse_sharding(raw_header["sharding"])
    validation = _parse_validation(raw_header["validation"])

    header = DatasetHeader(
        dataset_id=raw_header["dataset_id"],
        version=raw_header["version"],
        description=raw_header["description"],
        source_root=raw_header["source_root"],
        modalities=modalities,
        pairing=pairing,
        splits=splits,
        sharding=sharding,
        validation=validation,
    )

    assert header.dataset_id == "demo_ds"
    assert header.pairing.input_modality == "sat"
    assert header.splits.ratios == {"train": 0.7, "val": 0.2, "test": 0.1}
    assert header.sharding.shard_size == 8


def test_sharding_split_assignments(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    processed = tmp_path / "processed"
    dataset_root = extracted / "ms_buildings"
    sat_dir = dataset_root / "training" / "sat"
    map_dir = dataset_root / "training" / "map"

    for idx in range(10):
        img = np.full((3, 4, 4), idx, dtype=np.uint8)
        mask = np.full((1, 4, 4), idx % 2, dtype=np.uint8)
        _write_tiff(sat_dir / f"item_{idx}.tiff", img)
        _write_tiff(map_dir / f"item_{idx}.tif", mask)

    header = DatasetHeader(
        dataset_id="ms_buildings",
        version="0.0.1",
        description="test ms_buildings",
        source_root="ms_buildings",
        modalities=[
            ModalitySpec(name="sat", role="input", kind="raster", pattern="{split}/sat/*.tiff", channels=3, dtype="uint8"),
            ModalitySpec(name="map", role="target", kind="raster", pattern="{split}/map/*.tif", channels=1, dtype="uint8"),
        ],
        pairing=PairingSpec(strategy="by_stem", input_modality="sat", target_modality="map"),
        splits=SplitSpec(raw_splits=["training"], seed=123, labeled_policy="validation_only_with_metrics_subset", train_metric_fraction_of_labeled=0.25, ratios={"train": 0.6, "val": 0.3, "test": 0.1}, manifest_files=None),
        sharding=ShardingSpec(shard_size=3, layout_version=1, copy_mode="copy", force_uncompressed_tiff=True),
        validation=ValidationSpec(iou_ignore_label_leq=0, iou_average="macro_over_present_labels"),
        header_path=None,
    )

    summary = _build_for_header(header, extracted_root=extracted, output_root=processed, shard_size=None, seed_override=123, max_items=None, overwrite=True, dry_run=False)

    assert summary["train"]["items"] == 6
    assert summary["val"]["items"] == 3
    assert summary["test"]["items"] == 1

    shard_root = processed / "train" / "ms_buildings"
    entry, shard_dir = _read_first_index(shard_root)
    assert not Path(entry["input"]).is_absolute()
    assert entry["dataset_id"] == "ms_buildings"
    assert (shard_dir / entry["input"]).exists()


def test_processed_dataset_integrity(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    processed = tmp_path / "processed"
    dataset_root = extracted / "whu_building"
    train_dir = dataset_root / "train"

    for idx in range(10):
        img = np.full((3, 2, 2), idx, dtype=np.uint8)
        _write_tiff(train_dir / f"tile_{idx}.TIF", img)

    header = DatasetHeader(
        dataset_id="whu_building",
        version="0.0.1",
        description="test whu",
        source_root="whu_building",
        modalities=[ModalitySpec(name="sat", role="input", kind="raster", pattern="{split}/*.TIF", channels=3, dtype="uint8")],
        pairing=PairingSpec(strategy="by_stem", input_modality="sat", target_modality=None),
        splits=SplitSpec(raw_splits=["train", "validation", "test"], seed=123, labeled_policy="validation_only_with_metrics_subset", train_metric_fraction_of_labeled=0.0, ratios={"train": 0.6, "val": 0.3, "test": 0.1}, manifest_files=None),
        sharding=ShardingSpec(shard_size=5, layout_version=1, copy_mode="copy", force_uncompressed_tiff=True),
        validation=ValidationSpec(iou_ignore_label_leq=0, iou_average="macro_over_present_labels"),
        header_path=None,
    )

    summary = _build_for_header(header, extracted_root=extracted, output_root=processed, shard_size=None, seed_override=123, max_items=None, overwrite=True, dry_run=False)

    assert summary["train"]["items"] == 6
    assert summary["val"]["items"] == 3
    assert summary["test"]["items"] == 1

    shard_root = processed / "val" / "whu_building"
    entry, shard_dir = _read_first_index(shard_root)
    assert not Path(entry["input"]).is_absolute()

    input_path = shard_dir / entry["input"]
    assert input_path.exists()
    with rasterio.open(input_path) as ds:
        assert ds.count == 3
        assert ds.read().shape[1:] == (2, 2)


def test_inria_preserves_raw_split(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    processed = tmp_path / "processed"
    train_img_dir = extracted / "inria" / "train" / "images"
    train_gt_dir = extracted / "inria" / "train" / "gt"
    test_img_dir = extracted / "inria" / "test" / "images"

    _write_tiff(train_img_dir / "tile_a.tif", np.ones((3, 2, 2), dtype=np.uint8))
    _write_tiff(train_gt_dir / "tile_a.tif", np.zeros((1, 2, 2), dtype=np.uint8))
    _write_tiff(train_img_dir / "tile_b.tif", np.ones((3, 2, 2), dtype=np.uint8) * 2)
    _write_tiff(train_gt_dir / "tile_b.tif", np.ones((1, 2, 2), dtype=np.uint8) * 255)
    _write_tiff(test_img_dir / "tile_c.tif", np.ones((3, 2, 2), dtype=np.uint8) * 3)

    header = DatasetHeader(
        dataset_id="inria",
        version="0.0.1",
        description="inria test",
        source_root="inria",
        modalities=[
            ModalitySpec(name="sat", role="input", kind="raster", pattern="{split}/images/*.tif", channels=3, dtype="uint8"),
            ModalitySpec(name="map", role="target", kind="raster", pattern="{split}/gt/*.tif", channels=1, dtype="uint8"),
        ],
        pairing=PairingSpec(strategy="by_stem", input_modality="sat", target_modality="map"),
        splits=SplitSpec(raw_splits=["train", "test"], seed=123, labeled_policy="validation_only_with_metrics_subset", train_metric_fraction_of_labeled=0.0, ratios=None, manifest_files=None, preserve_raw_split=True),
        sharding=ShardingSpec(shard_size=1, layout_version=1, copy_mode="copy", force_uncompressed_tiff=True),
        validation=ValidationSpec(iou_ignore_label_leq=0, iou_average="macro_over_present_labels"),
        header_path=None,
    )

    summary = _build_for_header(header, extracted_root=extracted, output_root=processed, shard_size=None, seed_override=123, max_items=None, overwrite=True, dry_run=False)

    assert summary["train"]["items"] == 2
    assert summary["val"]["items"] == 0
    assert summary["test"]["items"] == 1