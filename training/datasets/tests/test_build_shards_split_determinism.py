# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

from pathlib import Path

import numpy as np
import pytest
import rasterio

from training.datasets.build_shards import _build_for_header
from training.datasets.header_schema import DatasetHeader, ModalitySpec, PairingSpec, ShardingSpec, SplitSpec, ValidationSpec


def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    profile = {"driver": "GTiff", "height": height, "width": width, "count": count, "dtype": array.dtype}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def _make_header(dataset_id: str, split_names: list[str]) -> DatasetHeader:
    return DatasetHeader(
        dataset_id=dataset_id,
        version="0.0.1",
        description=f"{dataset_id} demo",
        source_root=dataset_id,
        modalities=[
            ModalitySpec(name="sat", role="input", kind="raster", pattern="{split}/sat/*.tif", channels=3, dtype="uint8"),
            ModalitySpec(name="map", role="target", kind="raster", pattern="{split}/map/*.tif", channels=1, dtype="uint8"),
        ],
        pairing=PairingSpec(strategy="by_stem", input_modality="sat", target_modality="map"),
        splits=SplitSpec(raw_splits=split_names, seed=5, labeled_policy="validation_only_with_metrics_subset", train_metric_fraction_of_labeled=0.5, ratios={"train": 0.5, "val": 0.3, "test": 0.2}),
        sharding=ShardingSpec(shard_size=2, layout_version=1, copy_mode="copy", force_uncompressed_tiff=True),
        validation=ValidationSpec(iou_ignore_label_leq=0, iou_average="macro_over_present_labels"),
        header_path=None,
    )


def test_dry_run_counts_are_deterministic(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    dataset_root = extracted / "det"
    for idx in range(6):
        split = "train"
        _write_tiff(dataset_root / split / "sat" / f"sample_{idx}.tif", np.full((3, 2, 2), idx, dtype=np.uint8))
        _write_tiff(dataset_root / split / "map" / f"sample_{idx}.tif", np.full((1, 2, 2), idx % 2, dtype=np.uint8))

    header = _make_header("det", ["train"])
    counts_one = _build_for_header(header, extracted_root=extracted, output_root=tmp_path / "processed1", shard_size=None, seed_override=7, max_items=None, overwrite=True, dry_run=True)
    counts_two = _build_for_header(header, extracted_root=extracted, output_root=tmp_path / "processed2", shard_size=None, seed_override=7, max_items=None, overwrite=True, dry_run=True)

    assert counts_one == counts_two
    assert counts_one.get("train", 0) + counts_one.get("val", 0) + counts_one.get("test", 0) == 6


def test_overwrite_flag_prevents_clobbering(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    processed = tmp_path / "processed"
    dataset_root = extracted / "guard"
    _write_tiff(dataset_root / "train" / "sat" / "item.tif", np.ones((3, 2, 2), dtype=np.uint8))
    _write_tiff(dataset_root / "train" / "map" / "item.tif", np.ones((1, 2, 2), dtype=np.uint8))

    header = _make_header("guard", ["train"])
    _build_for_header(header, extracted_root=extracted, output_root=processed, shard_size=None, seed_override=1, max_items=None, overwrite=True, dry_run=False)

    with pytest.raises(FileExistsError):
        _build_for_header(header, extracted_root=extracted, output_root=processed, shard_size=None, seed_override=1, max_items=None, overwrite=False, dry_run=False)
