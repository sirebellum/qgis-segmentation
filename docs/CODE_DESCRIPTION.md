<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# CODE_DESCRIPTION

- Purpose: concise registry of modules and their current phase of stewardship (replaces CODE_SCRIPTURE.md).

## Runtime (Phase 0–6)
- segmenter.py / segmenter_dialog.py / segmenter_dialog_base.ui: QGIS UI + task dispatch for the single numpy runtime path; validates layer/segment count and queues `predict_nextgen_numpy`.
- funcs.py: numerical engine (materialize/tiling/stitching, cancellation/status helpers); includes `predict_nextgen_numpy` for the numpy runtime.
- qgis_funcs.py: GDAL render to GeoTIFF + layer registration.
- dependency_manager.py / perf_tuner.py / raster_utils.py: NumPy bootstrap, profiling shim, array utilities.
- model/runtime_numpy.py: numpy-only runtime for next-gen variable-K model consuming `model/best` artifacts (no torch import).
- model/README.md: artifact contract + producer/consumer notes for runtime.

## Training (Phase 3, scaffolding)
- training/config.py, config_loader.py: dataclass configs + python-loader overrides.
- training/data/: synthetic dataset placeholder with paired-view augmentations.
- training/models/: encoder stride/4, soft k-means head, refinement lanes, monolithic wrapper (later iterations removed elevation FiLM).
- training/losses.py, metrics.py: unsupervised objectives + proxy metrics.
- training/utils/: warp/gradients/resample/seed helpers.
- training/train.py, eval.py: CLI runners (synthetic-ready, eager PyTorch).
- training/tests/: pytest coverage (shapes, losses, synthetic smoke).

## Training (Phase 4, hardening)
- Elevation mask plumbing (since removed in Phase 13) and grad-accumulation tweaks.
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

## Docs (Phase 8)
- All supporting docs live under [docs/plugin](../plugin), [docs/training](../training), and [docs/dataset](.). History is tracked at [docs/AGENTIC_HISTORY.md](../AGENTIC_HISTORY.md); required prompt inputs listed in [docs/AGENTIC_REQUIRED.md](../AGENTIC_REQUIRED.md).

## Tests (Phase 8)
- [tests/test_alignment_invariants.py](../../tests/test_alignment_invariants.py): geotransform/bounds helpers (`derive_utm_epsg`, pixel sizes, tolerance).
- [tests/test_export_to_numpy_runtime.py](../../tests/test_export_to_numpy_runtime.py): export-to-runtime pipeline and probability normalization.
- [tests/test_numpy_runtime_tiling.py](../../tests/test_numpy_runtime_tiling.py): numpy runtime tiling/blending smoke via stub runtime.
- [tests/test_qgis_runtime_smoke.py](../../tests/test_qgis_runtime_smoke.py): optional QGIS import smoke (skips unless `QGIS_TESTS=1`).

## Notes
- TorchScript runtime paths were removed; plugin is numpy-only.
- Real raster IO is stubbed behind optional rasterio/gdal; synthetic paths remain the CI-safe default.

## Ops (Phase 10)
- Validation-only pass: python -m compileall . (pass) and ./.venv/bin/python -m pytest -q (44 passed, 1 skipped); no runtime changes.
- System python lacks pytest; use the repo venv for default test invocation to keep offline checks green.

## Ops (Phase 11 — NAIP AWS dry-run hardening)
- `prepare_naip_aws_3dep_dataset.py` now skips GDAL tool checks in `--dry-run` and uses an embedded NAIP index + DEM stub to avoid network failures; real runs unchanged.
- `scripts/data/_naip_aws_provider.py` gains GeoJSON-first querying, bbox overlap filtering, and stub index emission for dry-run.
- Added offline tests: `tests/test_prepare_naip_aws_3dep_dry_run.py` plus fixture `tests/fixtures/naip_index_min.geojson` to ensure dry-run succeeds without GDAL/network.

## Ops (Phase 12 — NAIP AWS real-run fallback)
- Added `--sample-data` smoke mode to `prepare_naip_aws_3dep_dataset.py` that downloads tiny public GitHub rasters (rgbsmall.tif + byte.tif) and executes the full warp/tile/manifest path without AWS/TNM credentials.
- Introduced explicit DEM override flags (`--dem-url`, `--dem-id`, `--dem-native-gsd`) and requester-pays headers for NAIP index downloads to avoid silent 403s.
- New offline tests cover DEM override short-circuit and NAIP requester-pays header propagation.

## Training (Phase 13 — RGB-only reset)
- Intent: remove elevation/DEM inputs while dataset ingestion is rewritten; keep runtime numpy path intact.
- Summary: reintroduced `training/data/` with RGB-only `SyntheticDataset` + `UnsupervisedRasterDataset`; dropped NAIP/3DEP manifest tests; model/loss/export/runtime paths no longer mention elevation metadata.
- Validation: pending full rerun of compileall/pytest after dataset tooling refresh; synthetic smoke remains default.

## Ops (Phase 14 — history reset + ingest scaffold)
- Added [docs/AGENTIC_HISTORY_SERIES_2.md](AGENTIC_HISTORY_SERIES_2.md) to start a new phase series with Phase 0 capturing the current numpy-only runtime, RGB-only training, and paused dataset ingestion.
- Appended pointer entry to [docs/AGENTIC_HISTORY.md](AGENTIC_HISTORY.md) while keeping prior phases intact.
- Seeded dataset ingestion scaffold under [scripts/datasets_ingest](../scripts/datasets_ingest) (config, interfaces, manifest validation, provider stubs, CLI); scaffold is offline-only and performs no network/GDAL work.
- Documented dataset rewrite status in [docs/dataset/DATASETS.md](dataset/DATASETS.md); added stub tests [tests/test_datasets_ingest_stub.py](../tests/test_datasets_ingest_stub.py) to keep coverage deterministic and QGIS-free.

## Ops (Phase 15 — runtime snapshot doc)
- Added [docs/plugin/RUNTIME_STATUS.md](plugin/RUNTIME_STATUS.md) as a token-efficient snapshot of the QGIS runtime (UI → task → numpy engine → render) with contracts, config points, and known gaps.
- Linked [docs/plugin/ARCHITECTURE.md](plugin/ARCHITECTURE.md) to the new snapshot and recorded the iteration in [docs/AGENTIC_HISTORY.md](AGENTIC_HISTORY.md).
