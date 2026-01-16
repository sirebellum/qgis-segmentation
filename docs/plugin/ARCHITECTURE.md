<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Architecture

For a current-state runtime snapshot, see [RUNTIME_STATUS.md](RUNTIME_STATUS.md). The plugin runtime has been restored to the original legacy path (TorchScript CNN + scikit-learn K-Means). Future integration of the new numpy/torch runtime is deferred until the new model is trained.

- Scope: QGIS 3 plugin "Map Segmenter"; dependency bootstrap runs on plugin import and installs torch, NumPy, and scikit-learn into `vendor/` unless skipped via env.

## End-to-end flow
- User opens the dialog from the plugin action in [segmenter.py](segmenter.py); UI defined by [segmenter_dialog_base.ui](segmenter_dialog_base.ui) and loaded via [segmenter_dialog.py](segmenter_dialog.py) (layer selector, segment count, smoothness/speed/accuracy sliders, log/progress, buttons).
- `Segmenter.run()` sets the device preference (CUDA → MPS → CPU) and initializes heuristics/progress state.
- `Segmenter.predict()` validates the selected GDAL raster (provider `gdal`, source `.tif/.tiff`, exactly 3 bands), parses segment count and resolution, builds blur + heuristic overrides, and dispatches a `QgsTask` via `run_task()`.
- Task executes either `legacy_kmeans_segmentation` or `legacy_cnn_segmentation` from [funcs.py](funcs.py) with a cancellation token + status callbacks. The CNN path loads a TorchScript model from [models/](../../models) (`model_<resolution>.pth`) and tiles/stitches via `predict_cnn`; the K-Means path clusters block samples via scikit-learn with optional torch-accelerated assignment.
- On completion, `Task.finished()` renders the uint8 label map to a temporary GeoTIFF with source extent/CRS through [qgis_funcs.py](qgis_funcs.py) `render_raster()` and adds it to the QGIS project; progress is finalized in the dialog.

## Module responsibilities
- [segmenter.py](segmenter.py): plugin entrypoint; menu/toolbar wiring; dialog lifecycle; progress/log handling; task wrapper; 3-band GeoTIFF validation; heuristic/blur tuning; dispatches legacy CNN/K-Means flows; device selection (CUDA → MPS → CPU).
- [segmenter_dialog.py](segmenter_dialog.py) / [segmenter_dialog_base.ui](segmenter_dialog_base.ui): Qt dialog shell and widgets (layer chooser, segment count, sliders, log/progress, buttons, logo hover interactions).
- [funcs.py](funcs.py): numerical engine for legacy flows. Raster materialization via GDAL, tiling/stitching, cancellation/status helpers, legacy CNN/TorchScript inference (`predict_cnn`) and legacy K-Means (`predict_kmeans`), optional GPU cluster assignment, blur smoothing.
- [qgis_funcs.py](qgis_funcs.py): writes numpy labels to temp GeoTIFF and registers a `QgsRasterLayer` with opacity 1.0.
- [dependency_manager.py](dependency_manager.py): on-demand vendor install of torch, NumPy, and scikit-learn (skip with `SEGMENTER_SKIP_AUTO_INSTALL`; torch index/args configurable via env); interpreter override via `SEGMENTER_PYTHON`.
- Adaptive settings: static defaults from [funcs.py](funcs.py) `AdaptiveSettings` (prefetch depth 2, safety factor 8); no device profiling thread.
- [raster_utils.py](raster_utils.py): `ensure_channel_first` helper for GDAL writes.
- [model/runtime_backend.py](model/runtime_backend.py), [model/runtime_numpy.py](model/runtime_numpy.py), [model/runtime_torch.py](model/runtime_torch.py): next-gen runtime selector and numpy/torch implementations for the new model type — **present but not wired into the current plugin flow; deferred until the new model is trained.**
- [model/README.md](model/README.md): artifact contract for the planned next-gen numpy runtime.
- [metadata.txt](metadata.txt): QGIS plugin metadata (name, version 2.2.1, author, tracker/homepage).

## Key extension points / config
- Runtime selection (current): user chooses CNN vs K-Means in the UI; device auto-selected (CUDA → MPS → CPU). Historical runtime selector env flags are inactive; next-gen runtime selector is deferred.
- Env toggles: `SEGMENTER_SKIP_AUTO_INSTALL`, `SEGMENTER_PYTHON`, torch index overrides (`SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`), and pip flags in `dependency_manager.py`.
- Rendering: `render_raster()` writes to tempdir GeoTIFF; layer name includes input layer, model choice, segment count, and resolution label.

## Runtime/perf notes (legacy path)
- Tiling: CNN path tiles to 512px by default with heuristic overrides (192–768px) and stitches back; K-Means uses block-based clustering and upsamples.
- Blur smoothing: optional legacy blur kernel/iteration applied to outputs for stability.
- Cancellation: `CancellationToken` checked throughout loops; task cancel toggles token and UI state.
- Logging/progress: worker status strings parsed by `_maybe_update_progress_from_message` keep the progress bar monotonic; log buffer shown in the dialog.

## Training scaffolding (isolated)
- Location: [training/](training) (eager PyTorch). Not wired into the current plugin runtime; exports numpy artifacts for the future runtime path.
- Components: config dataclasses, synthetic RGB dataset with paired-view augmentations, monolithic model (stride/4 encoder, soft k-means head, smoothing lanes), unsupervised losses, proxy metrics, CLI train/eval runners.
- Export: `training/export.py` writes best checkpoints to `model/best` and `training/best_model` as `model.npz` + `meta.json` + `metrics.json` when enabled.
- Note: Runtime update to consume these new artifacts is **deferred** until training is complete.
