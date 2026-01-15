<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# CODE_DESCRIPTION

- Purpose: concise registry of modules and their current phase of stewardship.

## Runtime (Phase 0–2, stable)
- segmenter.py / segmenter_dialog.py / segmenter_dialog_base.ui: QGIS UI + task dispatch; legacy K-Means/CNN inference only.
- funcs.py: numerical engine (tiling, clustering, latent KNN, blur); dependency/perf helpers live alongside.
- qgis_funcs.py: GDAL render to GeoTIFF + layer registration.
- dependency_manager.py / perf_tuner.py / raster_utils.py: bootstrap + profiling + array utilities.

## Training (Phase 3, scaffolding)
- training/config.py, config_loader.py: dataclass configs + python-loader overrides.
- training/data/: synthetic dataset, raster placeholder loader, paired-view augmentations.
- training/models/: encoder stride/4, elevation FiLM injection, soft k-means head, refinement lanes, monolithic wrapper.
- training/losses.py, metrics.py: unsupervised objectives + proxy metrics.
- training/utils/: warp/gradients/resample/seed helpers.
- training/train.py, eval.py: CLI runners (synthetic-ready, eager PyTorch).
- training/tests/: pytest coverage (shapes, losses, synthetic smoke).

## Notes
- TorchScript export and QGIS runtime integration are intentionally out-of-scope for Phase 3.
- Real raster IO is stubbed behind optional rasterio/gdal; synthetic paths remain the CI-safe default.
