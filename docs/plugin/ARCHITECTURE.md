<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Architecture

See [RUNTIME_STATUS.md](RUNTIME_STATUS.md) for a current-state runtime snapshot.

## Overview
QGIS 3 plugin "Map Segmenter" for unsupervised map segmentation using K-Means clustering. Dependency bootstrap runs on plugin import and installs torch + NumPy into `vendor/` unless skipped via env.

## End-to-end Flow
1. User opens dialog from plugin action in [segmenter.py](../../segmenter.py)
2. UI defined by [segmenter_dialog_base.ui](../../segmenter_dialog_base.ui), loaded via [segmenter_dialog.py](../../segmenter_dialog.py)
3. `Segmenter.run()` sets device preference (CUDA → MPS → CPU) and initializes state
4. `Segmenter.predict()` validates selected GDAL raster:
   - Provider must be `gdal`
   - Source must be `.tif/.tiff`
   - Must have exactly 3 bands
5. Parses segment count/resolution, builds heuristic overrides
6. Dispatches `QgsTask` via `run_task()` to run in background
7. Task executes `execute_kmeans_segmentation` from [runtime/pipeline.py](../../runtime/pipeline.py)
8. On completion, renders uint8 label map to GeoTIFF via [qgis_funcs.py](../../qgis_funcs.py)

## Module Responsibilities

### Plugin Core
- [segmenter.py](../../segmenter.py): plugin entrypoint, menu/toolbar wiring, dialog lifecycle, progress/log handling, task wrapper, heuristic tuning, device selection
- [segmenter_dialog.py](../../segmenter_dialog.py) / [segmenter_dialog_base.ui](../../segmenter_dialog_base.ui): Qt dialog shell and widgets

### Runtime Engine (runtime/)
- [pipeline.py](../../runtime/pipeline.py): high-level `execute_kmeans_segmentation`, optional blur
- [kmeans.py](../../runtime/kmeans.py): torch-only K-Means with global center fit + streaming assignment; implements seam prevention via halo overlap and globally-aligned block grid
- [chunking.py](../../runtime/chunking.py): chunk aggregation, label normalization, halo expansion utilities
- [adaptive.py](../../runtime/adaptive.py): chunk planning, memory budgets
- [distance.py](../../runtime/distance.py): pairwise distance kernels, chunked argmin
- [smoothing.py](../../runtime/smoothing.py): Gaussian blur channels
- [common.py](../../runtime/common.py): cancellation tokens, status callbacks, device coercion
- [io.py](../../runtime/io.py): raster materialization

### Support
- [funcs.py](../../funcs.py): compatibility facade re-exporting runtime/ symbols
- [qgis_funcs.py](../../qgis_funcs.py): GDAL GeoTIFF render + QgsRasterLayer registration
- [dependency_manager.py](../../dependency_manager.py): on-demand vendor install of torch/NumPy
- [raster_utils.py](../../raster_utils.py): `ensure_channel_first` helper
- [map_to_raster.py](../../map_to_raster.py): layer type detection + Convert map to raster dialog assist

## Key Contracts
- **Input validation**: 3-band GDAL GeoTIFF, enforced in `segmenter._is_supported_raster_layer`
- **Map-to-raster assist**: if user selects a web service or vector layer, `_on_layer_selection_changed` opens the Convert map to raster dialog prefilled with canvas extent + 1 map unit/pixel
- **K-Means**: torch-only, global center fit once, streaming assignment per chunk (no scikit-learn, no per-chunk relabeling)
- **Seam prevention**: block-level overlap (BLOCK_OVERLAP=1) with last-write-wins stitching, pixel halo (3px) for smoothing context, globally-aligned block grid, fixed float32 scaling
- **Device**: CUDA → MPS → CPU preference set in `segmenter.run()`
- **Cancellation**: `CancellationToken` checked throughout loops

## Config / Extension Points

### Environment Variables
- `SEGMENTER_SKIP_AUTO_INSTALL`: skip vendor dependency install
- `SEGMENTER_PYTHON`, `SEGMENTER_PIP_EXECUTABLE`: interpreter/pip overrides
- `SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`: torch wheel selection

### UI Controls
- Segment count: 2–16 typical
- Resolution: low/medium/high (maps to block sizes 16/8/4)
- Smoothing toggle: checkbox to enable/disable post-process boundary smoothing (default: disabled)
- Smoothness slider: 3 discrete levels (Low/Medium/High) with info icon — controls blur kernel size and iterations (only applies when smoothing enabled)

### Rendering
- `render_raster()` writes to tempdir GeoTIFF with source extent/CRS
- Layer name includes input layer, segment count, resolution

## Performance Notes
- **Global centers**: fitted once per run to ensure consistent labels across chunks
- **Streaming assignment**: labels consistent across chunk boundaries
- **Post-smoothing**: opt-in via checkbox; when disabled, segment boundaries are preserved exactly

## Security & Code Quality
Per QGIS plugin guidelines, the plugin is checked with:
- **bandit**: security vulnerability scanner
- **detect-secrets**: hardcoded credential detection
- **flake8**: Python linting

Run all checks:
```bash
.venv/bin/bandit -r __init__.py segmenter.py segmenter_dialog.py funcs.py qgis_funcs.py \
  dependency_manager.py raster_utils.py runtime/ --format txt
.venv/bin/detect-secrets scan __init__.py segmenter.py segmenter_dialog.py funcs.py \
  qgis_funcs.py dependency_manager.py raster_utils.py runtime/*.py
.venv/bin/flake8 __init__.py segmenter.py segmenter_dialog.py funcs.py qgis_funcs.py \
  dependency_manager.py raster_utils.py runtime/ --max-line-length=120 --ignore=E501,W503
```

See [RUNTIME_STATUS.md](RUNTIME_STATUS.md#security--linting-qgis-recommended) for detailed expectations.

## Deprecated / Removed

### CNN Runtime (removed as of Phase 2)
The CNN inference path has been removed from the plugin runtime:
- No model dropdown in UI
- No TorchScript model loading
- No CNN tiling/inference
- The `models/` directory and `runtime/cnn.py` are not used by the runtime segmentation path

### Training Pipeline (not shipped)
Training code in [training/](../../training) is isolated from the plugin runtime and is not part of the distributed plugin.
