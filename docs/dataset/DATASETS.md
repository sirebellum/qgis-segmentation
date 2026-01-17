<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Datasets (Rewrite In Progress)

- Status: Dataset ingestion was redesigned; historical NAIP/DEM loaders were removed. Training now supports shard-backed ingestion (`data.source=shards`) alongside the synthetic default.
- New header + shard pipeline (v0):
	- Headers live in [training/datasets/headers](../../training/datasets/headers) and follow the schema in [docs/dataset/HEADERS.md](HEADERS.md).
	- Generate headers from extracted data: `python -m training.datasets.generate_headers --dataset ms_buildings --dry-run` (inspect) then rerun without `--dry-run`.
	- Build uncompressed GeoTIFF shards + `index.jsonl`: `python -m training.datasets.build_shards --dataset-id ms_buildings --overwrite --seed 123 --shard-size 512` (outputs under `training/datasets/processed/<dataset>/<split>/shard-xxxxx`).
	- Labeled items are held out for validation; 25% of labeled tiles feed `metrics_train` (metrics-only) and the rest go to `val`. Unlabeled tiles go to `train`.
- Shard contract consumed by the trainer:
	- Layout: `training/datasets/processed/<dataset_id>/<split>/shard-xxxxx/` containing `inputs/*.tif`, optional `targets/*.tif`, and `index.jsonl`.
	- `index.jsonl` fields: `dataset_id`, `item_id`, `raw_split`, normalized `split`, `input` (relative path), `has_target`, and optional `target` (relative path). Paths must be relative to the shard directory.
	- Metrics: targets are ignored by the loss and used only for IoU evaluation; labels `<=0` are masked out so unlabeled/background pixels do not affect scores.
- Prototype dataset (present in repo): `training/datasets/extracted/ms_buildings` with `sat/*.tiff` (RGB) paired to `map/*.tif` (binary building masks) across `training/val/test` splits. The generated header locks patterns, channel/dtype hints, and IoU ignore rules (`label<=0` ignored). GeoTIFF patches (512x512) are now the primary training input via `training/data/geo_patch_dataset.py`; synthetic data remains a smoke path.
- Manifest scaffold (unchanged): [scripts/datasets_ingest/README.md](../../scripts/datasets_ingest/README.md) documents the offline-only manifest stubs and CLI; still stub-only and network/GDAL-free.
- Legacy artifacts: sample configs under [configs/datasets](../../configs/datasets) remain for reference but are not currently wired into the training loaders. GDAL helpers persist in [scripts/data/_gdal_utils.py](../../scripts/data/_gdal_utils.py) for future reuse.
- Tests: default pytest remains QGIS-free and offline; new dataset tests cover header parsing, shard layout, split determinism, and IoU masking.
