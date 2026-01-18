# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

from pathlib import Path

import pytest
import yaml

from training.datasets.header_schema import _parse_modality, _parse_pairing, _parse_splits, load_header


def test_invalid_modality_role_raises():
    raw = {"name": "sat", "role": "unknown", "kind": "raster", "pattern": "*.tif"}
    with pytest.raises(ValueError):
        _parse_modality(raw)


def test_pairing_requires_known_modalities():
    modalities = [_parse_modality({"name": "sat", "role": "input", "kind": "raster", "pattern": "*.tif"})]
    with pytest.raises(ValueError):
        _parse_pairing({"strategy": "by_stem", "input_modality": "map", "target_modality": "map"}, modalities)


def test_split_ratio_validation():
    with pytest.raises(ValueError):
        _parse_splits({"raw": ["train"], "ratios": {"train": -0.1}})


def test_load_header_minimum_fields(tmp_path: Path):
    header_path = tmp_path / "header.yaml"
    payload = {
        "dataset_id": "toy",
        "version": "0.0.1",
        "description": "demo header",
        "source_root": "toy",
        "modalities": [
            {"name": "sat", "role": "input", "kind": "raster", "pattern": "{split}/sat/*.tif", "channels": 3},
        ],
        "pairing": {"strategy": "by_stem", "input_modality": "sat"},
        "splits": {"raw": ["train"], "ratios": {"train": 1.0}},
        "sharding": {"shard_size": 2},
        "validation": {"iou_ignore_label_leq": 0},
    }
    header_path.write_text("# SPDX-License-Identifier: BSD-3-Clause\n" + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    header = load_header(header_path)
    assert header.dataset_id == "toy"
    assert header.modalities[0].channels == 3
    assert header.splits.ratios == {"train": 1.0}
