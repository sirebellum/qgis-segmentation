<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Architecture

For a current-state runtime snapshot, see [RUNTIME_STATUS.md](RUNTIME_STATUS.md).

- Scope: QGIS 3 plugin "Map Segmenter"; code is source of truth. Dependency bootstrap runs on plugin load and now optionally installs torch when enabled.

## End-to-end flow
- User opens dialog from plugin action in [segmenter.py](segmenter.py); UI defined by [segmenter_dialog_base.ui](segmenter_dialog_base.ui) and loaded via [segmenter_dialog.py](segmenter_dialog.py).
- `Segmenter.predict()` validates 3-band GDAL raster selection, parses segment count, and dispatches a `QgsTask` via `run_task()`.
- Task runs `predict_nextgen_numpy()` from [funcs.py](funcs.py) with cancellation token + progress callbacks while the runtime loader in [model/runtime_backend.py](model/runtime_backend.py) selects torch (CUDA/MPS/CPU) when available or falls back to numpy. Raster IO stays off the UI thread.
- Segmentation output is rendered to a temporary GeoTIFF with source extent/CRS through [qgis_funcs.py](qgis_funcs.py) `render_raster()` and added to the QGIS project.

## Module responsibilities
- [segmenter.py](segmenter.py): plugin entrypoint; menu/toolbar wiring; dialog lifecycle; progress bar/status logging; `QgsTask` wrapper; loads numpy runtime from `model/best` and dispatches `predict_nextgen_numpy`; enforces GDAL + `.tif/.tiff` + 3-band requirement with user-facing reasons.
- [segmenter_dialog.py](segmenter_dialog.py) / [segmenter_dialog_base.ui](segmenter_dialog_base.ui): Qt dialog shell and widgets (layer selector, segment count input, log/progress, buttons).
- [funcs.py](funcs.py): numerical engine. Raster materialization (strict 3-band channel-first validation + `.tif/.tiff` check), tiling/stitching, cancellation/status helpers, and tiled `predict_nextgen_numpy` that accepts any backend exposing `predict_labels`.
- [model/runtime_backend.py](model/runtime_backend.py): backend selector (torch preferred, numpy fallback) with device preference handling; logs fallback reason via status callbacks.
- [model/runtime_torch.py](model/runtime_torch.py): torch runtime consuming the same `model.npz` + `meta.json` artifacts, preferring CUDA → MPS → CPU; mirrors numpy forward pass.
- [qgis_funcs.py](qgis_funcs.py): writes numpy labels to temp GeoTIFF and registers a `QgsRasterLayer` with opacity preserved.
- [dependency_manager.py](dependency_manager.py): on-demand vendor install of NumPy into `vendor/`, honoring env toggles (`SEGMENTER_*`); optional torch install is gated by `SEGMENTER_ENABLE_TORCH=1` and otherwise skipped.
- [perf_tuner.py](perf_tuner.py): compatibility shim returning default adaptive settings (no profiling).
- [raster_utils.py](raster_utils.py): `ensure_channel_first` utility for GDAL writes.
- [model/runtime_numpy.py](model/runtime_numpy.py): numpy-only runtime for the next-gen variable-K model consuming `model/best` artifacts.
- [model/README.md](model/README.md): artifact contract and producer/consumer notes for the numpy runtime.
- [metadata.txt](metadata.txt): QGIS plugin metadata (name, version 2.2.1, author, tracker/homepage).

## Key extension points / config
- Runtime: backend selector prefers torch GPU/CPU when available, falls back to numpy; segment count is user-configurable; torch path remains optional.
- Env toggles: `SEGMENTER_SKIP_AUTO_INSTALL`, `SEGMENTER_PYTHON` for bootstrap; `SEGMENTER_ENABLE_TORCH` to attempt installing torch; `SEGMENTER_RUNTIME_BACKEND`/`SEGMENTER_DEVICE` override backend/device selection.
- Rendering: `render_raster()` uses tempdir GeoTIFF; layer name includes input layer and segment count.

## Runtime/perf notes
- Tiling: `tile_raster` pads to square tiles, stitches canvas back, and trims padding after inference. Inputs must already be validated 3-band arrays.
- Cancellation: `CancellationToken` checked via `_maybe_raise_cancel` across loops; task cancel cancels token and updates UI.
- Logging/progress: worker emits status strings parsed by `_maybe_update_progress_from_message` to keep UI progress bar moving; history buffered in dialog log.

## Training scaffolding (isolated)
- Location: [training/](training) (pure PyTorch, eager mode). Not wired into QGIS runtime; exports numpy artifacts for runtime pickup.
- Components: config dataclasses, synthetic RGB dataset with paired-view augmentations, monolithic model (encoder stride/4, soft k-means head, fast/learned refinement lanes), unsupervised losses (consistency, entropy shaping, edge-aware smoothness), proxy metrics, CLI train/eval runners.
- Export: `training/export.py` writes best checkpoint weights to `model/best` and `training/best_model` as `model.npz` + `meta.json` + `metrics.json` when training loss improves (unless `--no-export`).
- Contract: forward(rgb, K) → probabilities [B,K,512,512] (K ≤ 16) + embeddings stride/4; RGB-only inputs.
- CLI: `python -m training.train --steps 3 --grad-accum 2 --amp 0` for smoke training; `python -m training.eval --synthetic` for proxy metrics.
