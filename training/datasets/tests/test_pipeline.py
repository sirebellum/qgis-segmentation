# SPDX-License-Identifier: BSD-3-Clause
import json
import random
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml

from training.datasets.build_shards import _build_for_header
from training.datasets.header_schema import DatasetHeader, ModalitySpec, PairingSpec, ShardingSpec, SplitSpec, ValidationSpec, load_header
from training.datasets.metrics import masked_iou


def _write_tiff(path: Path, array: np.ndarray, compress: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": array.dtype,
    }
    if compress:
        profile["compress"] = compress
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def test_header_schema_requires_fields(tmp_path: Path) -> None:
    bad_header = tmp_path / "bad.yaml"
    bad_header.write_text("source_root: nowhere\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_header(bad_header)

    good_header = tmp_path / "good.yaml"
    payload = {
        "dataset_id": "toy",
        "version": "0.0.1",
        "description": "toy dataset",
        "source_root": "toy",
        "modalities": [
            {"name": "sat", "role": "input", "kind": "raster", "pattern": "{split}/sat/*.tif"}
        ],
        "pairing": {"strategy": "by_stem", "input_modality": "sat"},
        "splits": {"raw": ["demo"], "seed": 1},
        "sharding": {"shard_size": 2},
        "validation_metrics": {"iou_ignore_label_leq": 0},
    }
    good_header.write_text("# SPDX-License-Identifier: BSD-3-Clause\n" + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    header = load_header(good_header)
    assert header.dataset_id == "toy"
    assert header.splits.raw_splits == ["demo"]
    assert header.modalities[0].pattern.startswith("{split}/sat")


def test_shard_builder_layout_and_split(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    processed = tmp_path / "processed"
    dataset_root = extracted / "toy"
    split_name = "demo"
    sat_dir = dataset_root / split_name / "sat"
    map_dir = dataset_root / split_name / "map"

    rng = np.random.default_rng(0)
    # labeled items
    for idx in range(2):
        array = rng.integers(0, 255, size=(3, 4, 4), dtype=np.uint8)
        _write_tiff(sat_dir / f"labeled_{idx}.tif", array, compress="lzw")
        mask = np.zeros((3, 4, 4), dtype=np.uint8)
        mask[:, :2, :2] = 255
        _write_tiff(map_dir / f"labeled_{idx}.tif", mask)
    # unlabeled item (no target)
    _write_tiff(sat_dir / "unlabeled.tif", np.ones((3, 4, 4), dtype=np.uint8) * 10)

    header = DatasetHeader(
        dataset_id="toy",
        version="0.0.1",
        description="toy dataset",
        source_root="toy",
        modalities=[
            ModalitySpec(name="sat", role="input", kind="raster", pattern="{split}/sat/*.tif", channels=3, dtype="uint8"),
            ModalitySpec(name="map", role="target", kind="raster", pattern="{split}/map/*.tif", channels=3, dtype="uint8"),
        ],
        pairing=PairingSpec(strategy="by_stem", input_modality="sat", target_modality="map"),
        splits=SplitSpec(raw_splits=[split_name], seed=7, labeled_policy="validation_only_with_metrics_subset", train_metric_fraction_of_labeled=0.5),
        sharding=ShardingSpec(shard_size=1, layout_version=1, copy_mode="copy", force_uncompressed_tiff=True),
        validation=ValidationSpec(iou_ignore_label_leq=0, iou_average="macro_over_present_labels"),
        header_path=None,
    )

    summary = _build_for_header(
        header,
        extracted_root=extracted,
        output_root=processed,
        shard_size=None,
        seed_override=7,
        max_items=None,
        overwrite=True,
        dry_run=False,
    )

    assert summary["train"]["items"] == 2  # unlabeled + metrics_train folded into train
    assert summary["val"]["items"] == 1
    assert summary["test"]["items"] == 0

    summary_path = processed / "metadata" / "toy_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["dataset_id"] == "toy"
    assert payload["layout_version"] == 1

    # verify compress is none in outputs
    shard_dirs = list((processed / "train" / "toy").glob("shard-*")) + list((processed / "val" / "toy").glob("shard-*"))
    assert shard_dirs
    for shard_dir in shard_dirs:
        for tif_path in shard_dir.rglob("*.tif"):
            with rasterio.open(tif_path) as src:
                assert src.profile.get("compress") is None

    # verify index entries reference relative paths
    shard_root = processed / "train" / "toy"
    sample_index = next(shard_root.glob("**/index.jsonl"))
    lines = sample_index.read_text().strip().splitlines()
    assert lines
    entry = json.loads(lines[0])
    assert not Path(entry["input"]).is_absolute()


def test_masked_iou_ignores_background() -> None:
    pred = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)
    gt = np.array([[0, 1, 2], [2, 0, 0]], dtype=np.uint8)
    result = masked_iou(pred, gt, ignore_label_leq=0)
    assert result["has_valid_labels"] is True
    per_class = result["per_class"]
    assert 0 not in per_class
    assert set(per_class.keys()) == {1, 2}
    # label 1 intersection=1, union=2 => 0.5
    assert per_class[1] == pytest.approx(0.5)


def test_ratio_split_assignment(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    processed = tmp_path / "processed"
    dataset_root = extracted / "ratio"
    split_name = "raw"
    sat_dir = dataset_root / split_name / "sat"

    for idx in range(10):
        array = np.full((3, 2, 2), idx, dtype=np.uint8)
        _write_tiff(sat_dir / f"item_{idx}.tif", array)

    header = DatasetHeader(
        dataset_id="ratio",
        version="0.0.1",
        description="ratio dataset",
        source_root="ratio",
        modalities=[
            ModalitySpec(name="sat", role="input", kind="raster", pattern="{split}/sat/*.tif", channels=3, dtype="uint8"),
        ],
        pairing=PairingSpec(strategy="by_stem", input_modality="sat", target_modality=None),
        splits=SplitSpec(raw_splits=[split_name], seed=42, labeled_policy="validation_only_with_metrics_subset", train_metric_fraction_of_labeled=0.0, ratios={"train": 0.6, "val": 0.3, "test": 0.1}),
        sharding=ShardingSpec(shard_size=10, layout_version=1, copy_mode="copy", force_uncompressed_tiff=True),
        validation=ValidationSpec(iou_ignore_label_leq=0, iou_average="macro_over_present_labels"),
        header_path=None,
    )

    counts = _build_for_header(
        header,
        extracted_root=extracted,
        output_root=processed,
        shard_size=None,
        seed_override=None,
        max_items=None,
        overwrite=True,
        dry_run=True,
    )

    assert counts["train"] == 6
    assert counts["val"] == 3
    assert counts["test"] == 1
