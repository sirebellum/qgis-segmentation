<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# CODE_DESCRIPTION

- Purpose: concise registry of modules and their current phase of stewardship (replaces CODE_SCRIPTURE.md).

## Runtime (Phase 0–6)
- segmenter.py / segmenter_dialog.py / segmenter_dialog_base.ui: QGIS UI + task dispatch; legacy K-Means/CNN inference; routes "Next-Gen (Numpy)" option to runtime loader.
- funcs.py: numerical engine (tiling, clustering, latent KNN, blur); dependency/perf helpers; includes `predict_nextgen_numpy` for numpy runtime.
- qgis_funcs.py: GDAL render to GeoTIFF + layer registration.
- dependency_manager.py / perf_tuner.py / raster_utils.py: bootstrap + profiling + array utilities.
- model/runtime_numpy.py: numpy-only runtime for next-gen variable-K model consuming `model/best` artifacts (no torch import).
- model/README.md: artifact contract + producer/consumer notes for runtime.

## Training (Phase 3, scaffolding)
- training/config.py, config_loader.py: dataclass configs + python-loader overrides.
- training/data/: synthetic dataset, raster placeholder loader, paired-view augmentations.
- training/models/: encoder stride/4, elevation FiLM injection, soft k-means head, refinement lanes, monolithic wrapper.
- training/losses.py, metrics.py: unsupervised objectives + proxy metrics.
- training/utils/: warp/gradients/resample/seed helpers.
- training/train.py, eval.py: CLI runners (synthetic-ready, eager PyTorch).
- training/tests/: pytest coverage (shapes, losses, synthetic smoke).

## Training (Phase 4, hardening)
- Elevation masks threaded through collate → FiLM gate; dropout preserved.
- Fast smoothing fixed to depthwise kernel for arbitrary K; optional grad accumulation exposed in CLI/train loop.
- Synthetic-first default retained; real raster IO still stubbed.
- Docs aligned: ARCHITECTURE.md, MODEL_NOTES.md, training/README.md, training/MODEL_HISTORY.md.

## Training (Phase 5, NAIP/3DEP scaffolding)
- scripts/data/prepare_naip_3dep_dataset.py + helpers: TNM discovery, GDAL-based mosaic/clip/reproject, DEM-to-RGB alignment, tiling, manifest.jsonl.
- training/data/naip_3dep_dataset.py: manifest reader + elevation standardization over rasterio; reusable for other manifest-based paired rasters.
- configs/datasets/naip_3dep_example.yaml: sample manifest-driven config (PyYAML optional, python configs still supported).
- Docs added: docs/DATASETS.md, docs/TRAINING_PIPELINE.md.

## Training (Phase 7, NAIP-on-AWS refactor)
- scripts/data/prepare_naip_aws_3dep_dataset.py: NAIP AWS Requester Pays COG ingestion via cached footprint index + 3DEP TNM Access DEM ladder; derives a single grid (-tap) and warps DEM to RGB grid; tiles 512x512/stride 128 with nodata filtering; emits enriched manifest (tier/native/resampled metadata).
- scripts/data/_naip_aws_provider.py: NAIP AWS bucket/index wrapper (footprint download, vsis3/HTTPS VRT build).
- scripts/data/_usgs_3dep_provider.py: TNM Access DEM discovery with tier ladder (1m → 1/3" → 1").
- training/data/naip_aws_3dep_dataset.py: thin alias to shared manifest loader; loader accepts extended manifest fields (nodata_fraction, dem_native_gsd, resampled flag, source_naip_urls).
- configs/datasets/naip_aws_3dep_example.yaml: sample config pointing at `data/naip_aws_3dep/processed/manifest.jsonl`.

## Training (Phase 6, export to runtime)
- training/export.py: converts MonolithicSegmenter checkpoints to numpy artifacts (`model.npz`, `meta.json`, `metrics.json`).
- training/train.py: tracks best loss and auto-exports to `model/best` and `training/best_model` (configurable; can disable with `--no-export`).
- func_test.py: includes dummy runtime smoke test for numpy loader.

## Notes
- TorchScript export remains optional; legacy CNN path still uses TorchScript weights if present.
- Real raster IO is stubbed behind optional rasterio/gdal; synthetic paths remain the CI-safe default.
