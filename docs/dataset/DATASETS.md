<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Datasets

## Status
Dataset ingestion uses a header + shard pipeline. Training supports shard-backed ingestion (`data.source=shards`) alongside the synthetic default.

## Header + Shard Pipeline

### Headers
- Location: [training/datasets/headers/](../../training/datasets/headers)
- Schema: [docs/dataset/HEADERS.md](HEADERS.md)
- Available headers: `ms_buildings.yaml`, `whu_building.yaml`, `openearth.yaml`, `inria.yaml`

### Generate Headers
```bash
# Inspect first
python -m training.datasets.generate_headers --dataset ms_buildings --dry-run

# Then write
python -m training.datasets.generate_headers --dataset ms_buildings
```

### Build Shards
```bash
python -m training.datasets.build_shards --dataset-id ms_buildings --overwrite --seed 123 --shard-size 512
```
Outputs to `training/datasets/processed/<dataset>/<split>/shard-xxxxx/`.

### Split Policy
- Labeled items held out for validation
- 25% of labeled tiles → `metrics_train` (metrics-only)
- Remaining labeled → `val`
- Unlabeled tiles → `train`

### Pairing Policy
Headers accept:
- `pairing.on_missing_input`: `drop_item` | `error` (default: `drop_item`)
- `pairing.on_missing_target`: `allow` | `drop_item` | `error` (default: `allow`)
- `pairing.strategy=regex` with `pairing.stem_regex` for custom stem extraction

## Shard Contract

### Layout
```
training/datasets/processed/<dataset_id>/<split>/shard-xxxxx/
├── inputs/*.tif
├── targets/*.tif (optional)
├── slic/*.npz
└── index.jsonl
```

### index.jsonl Fields
- `dataset_id`, `item_id`, `raw_split`, `split`
- `input`: relative path to input tile
- `has_target`: boolean
- `target`: relative path (if present)
- `slic`: relative path to SLIC labels

### Metrics
- Targets used only for IoU evaluation (not in loss)
- Labels ≤0 masked during IoU computation

## SLIC Precompute
- Superpixels computed during shard build (not at training time)
- `data.require_slic=true` by default in trainer
- Fallback to grid if OpenCV contrib unavailable

## Data Sources (Training)

### Synthetic (default)
```bash
python -m training.train --synthetic --steps 3 --seed 123
```
Random RGB tensors for offline/CI testing.

### Shard-backed
Set `data.source=shards` in config or via CLI.

### GeoTIFF Patches
Random windows via `training/data/geo_patch_dataset.py`.

## Tests
```bash
.venv/bin/python -m pytest training/tests/ -q
```
Covers header parsing, shard layout, split determinism, and IoU masking.
