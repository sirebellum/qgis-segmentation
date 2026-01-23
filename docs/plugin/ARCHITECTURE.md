<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Architecture

See [RUNTIME_STATUS.md](RUNTIME_STATUS.md) for a current-state runtime snapshot.

## Overview
QGIS 3 plugin "Map Segmenter" for unsupervised map segmentation. Dependency bootstrap runs on plugin import and installs torch + NumPy into `vendor/` unless skipped via env.

## End-to-end Flow
1. User opens dialog from plugin action in [segmenter.py](../../segmenter.py)
2. UI defined by [segmenter_dialog_base.ui](../../segmenter_dialog_base.ui), loaded via [segmenter_dialog.py](../../segmenter_dialog.py)
3. `Segmenter.run()` sets device preference (CUDA → MPS → CPU) and initializes state
4. `Segmenter.predict()` validates selected GDAL raster:
   - Provider must be `gdal`
   - Source must be `.tif/.tiff`
   - Must have exactly 3 bands
5. Parses segment count/resolution, builds blur + heuristic overrides
6. Dispatches `QgsTask` via `run_task()` to run in background
7. Task executes `execute_kmeans_segmentation` or `execute_cnn_segmentation` from [runtime/pipeline.py](../../runtime/pipeline.py)
8. On completion, renders uint8 label map to GeoTIFF via [qgis_funcs.py](../../qgis_funcs.py)

## Module Responsibilities

### Plugin Core
- [segmenter.py](../../segmenter.py): plugin entrypoint, menu/toolbar wiring, dialog lifecycle, progress/log handling, task wrapper, heuristic tuning, device selection
- [segmenter_dialog.py](../../segmenter_dialog.py) / [segmenter_dialog_base.ui](../../segmenter_dialog_base.ui): Qt dialog shell and widgets

### Runtime Engine (runtime/)
- [pipeline.py](../../runtime/pipeline.py): high-level `execute_cnn_segmentation`, `execute_kmeans_segmentation`, optional blur, texture autoencoder hooks
- [kmeans.py](../../runtime/kmeans.py): torch-only K-Means with global center fit + streaming assignment
- [cnn.py](../../runtime/cnn.py): CNN tiling, prefetch, global center fit, inference with centers
- [chunking.py](../../runtime/chunking.py): chunk aggregation, label normalization, one-hot conversion
- [adaptive.py](../../runtime/adaptive.py): chunk planning, memory budgets, tile sizing
- [distance.py](../../runtime/distance.py): pairwise distance kernels, chunked argmin
- [smoothing.py](../../runtime/smoothing.py): Gaussian blur channels
- [latent.py](../../runtime/latent.py): latent KNN soft refinement
- [common.py](../../runtime/common.py): cancellation tokens, status callbacks, device coercion
- [io.py](../../runtime/io.py): raster/model materialization

### Support
- [funcs.py](../../funcs.py): compatibility facade re-exporting runtime/ symbols
- [qgis_funcs.py](../../qgis_funcs.py): GDAL GeoTIFF render + QgsRasterLayer registration
- [dependency_manager.py](../../dependency_manager.py): on-demand vendor install of torch/NumPy
- [raster_utils.py](../../raster_utils.py): `ensure_channel_first` helper
- [autoencoder_utils.py](../../autoencoder_utils.py): texture autoencoder manager (optional, disabled by default)

### Model Artifacts
- [models/](../../models): TorchScript CNN weights (`model_4.pth`, `model_8.pth`, `model_16.pth`)

## Key Contracts
- **Input validation**: 3-band GDAL GeoTIFF, enforced in `segmenter._is_supported_raster_layer`
- **K-Means**: torch-only, global center fit once, streaming assignment per chunk (no scikit-learn, no per-chunk relabeling)
- **CNN**: TorchScript models, global centers + chunked assignment via `_process_in_chunks`
- **Device**: CUDA → MPS → CPU preference set in `segmenter.run()`
- **Cancellation**: `CancellationToken` checked throughout loops

## Config / Extension Points

### Environment Variables
- `SEGMENTER_SKIP_AUTO_INSTALL`: skip vendor dependency install
- `SEGMENTER_PYTHON`, `SEGMENTER_PIP_EXECUTABLE`: interpreter/pip overrides
- `SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`: torch wheel selection

### UI Controls
- Model choice: CNN vs K-Means
- Segment count: 2–16 typical
- Resolution: low/medium/high (maps to model_16/8/4.pth)
- Sliders: smoothness, speed, accuracy (tune blur kernel, tile size, sampling scale)

### Rendering
- `render_raster()` writes to tempdir GeoTIFF with source extent/CRS
- Layer name includes input layer, model choice, segment count, resolution

## Performance Notes
- **Tiling**: CNN tiles to 512px by default (192–768px with heuristics)
- **Global centers**: fitted once per run to ensure consistent labels across chunks
- **Post-smoothing**: optional blur kernel/iterations on outputs
- **Prefetch**: adaptive batch sizing based on memory budget

## Security & Code Quality
Per QGIS plugin guidelines, the plugin is checked with:
- **bandit**: security vulnerability scanner
- **detect-secrets**: hardcoded credential detection
- **flake8**: Python linting

Run all checks:
```bash
.venv/bin/bandit -r __init__.py segmenter.py segmenter_dialog.py funcs.py qgis_funcs.py \
  dependency_manager.py raster_utils.py autoencoder_utils.py runtime/ --format txt
.venv/bin/detect-secrets scan *.py runtime/*.py
.venv/bin/flake8 *.py runtime/ --max-line-length=120 --ignore=E501,W503
```

See [RUNTIME_STATUS.md](RUNTIME_STATUS.md#security--linting-qgis-recommended) for detailed expectations.

## Training (Isolated)
Training code in [training/](../../training) is isolated from the plugin runtime:
- Produces student embeddings via distillation
- Exports numpy artifacts (not yet consumed by plugin)
- See [training/README.md](../../training/README.md) for details
