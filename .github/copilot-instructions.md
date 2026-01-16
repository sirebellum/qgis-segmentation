# QGIS Segmenter – Copilot Instructions

## Architecture & flow
- UI entry: [../segmenter.py](../segmenter.py) `Segmenter` class builds the dialog from [../segmenter_dialog_base.ui](../segmenter_dialog_base.ui) via [../segmenter_dialog.py](../segmenter_dialog.py); buttons call `predict()`/`stop_current_task()`.
- Task dispatch: `Segmenter.predict()` validates GDAL `.tif/.tiff` 3-band raster selection and segment count, then enqueues a `QgsTask` via `run_task()` to keep the UI thread idle.
- Runtime path: task calls [../funcs.py](../funcs.py) `predict_nextgen_numpy()`; backend selection happens in [../model/runtime_backend.py](../model/runtime_backend.py) (torch preferred if available and enabled, otherwise numpy). Both backends expose `predict_labels` consumed by tiling/stitching in `funcs.py`.
- Rendering: [../qgis_funcs.py](../qgis_funcs.py) `render_raster()` writes temp GeoTIFF with source extent/CRS and re-adds as `QgsRasterLayer` (opacity 1.0).

## Dependencies & env toggles
- Auto-install: [../dependency_manager.py](../dependency_manager.py) `ensure_dependencies()` vendors NumPy into `vendor/` at plugin load unless `SEGMENTER_SKIP_AUTO_INSTALL=1`.
- Optional torch: set `SEGMENTER_ENABLE_TORCH=1` to attempt torch install; otherwise torch is only used if already present. Override interpreter with `SEGMENTER_PYTHON`; pip override via `SEGMENTER_PIP_EXECUTABLE`.
- Runtime overrides: `SEGMENTER_RUNTIME_BACKEND` (`auto`/`torch`/`numpy`) and `SEGMENTER_DEVICE` (`auto`/`cuda`/`mps`/`cpu`) steer backend/device selection.

## Models & data contract
- Artifacts: [../model/best](../model/README.md) ships `model.npz` + `meta.json`; runtime assumes channel-first RGB, uint8/float32 compatible input, and returns uint8 labels.
- Input enforcement: `_is_supported_raster_layer()` in [../segmenter.py](../segmenter.py) and `_materialize_raster()` in [../funcs.py](../funcs.py) require `.tif/.tiff`, provider `gdal`, exactly 3 bands; errors are user-facing.
- Tile/stitch: `tile_raster()` pads to `TILE_SIZE` (256), stitches canvas, then trims padding.

## UI/status patterns
- Progress: worker status strings parsed by `_maybe_update_progress_from_message()` to advance the bar; stage bounds in `PROGRESS_STAGE_MAP` ensure monotonic bar updates.
- Cancellation: `CancellationToken` bound to `QgsTask`; call `_maybe_raise_cancel()` inside loops (already in `predict_nextgen_numpy`) and let `Task.cancel()` propagate.
- Logging: `Segmenter.log_status()` for UI-safe messages; worker threads should use `status_callback` passed into `predict_nextgen_numpy()`.

## Testing & workflows
- Runtime tests: `pytest func_test.py` plus focused markers (`-k predict`, `-m performance`). Training tests live under [../training/tests](../training/tests) and can be run with `pytest training/tests`.
- Packaging: `pb_tool zip` to build the distributable; `pb_tool deploy` (see [../pb_tool.cfg](../pb_tool.cfg)) to sync into a QGIS profile.
- Qt assets: after editing `resources.qrc`, run `pyrcc5 resources.qrc -o resources.py`. If the UI file changes, re-run `pyuic5 segmenter_dialog_base.ui -o segmenter_dialog.py` and keep widget names stable.

## Training (isolated from plugin runtime)
- Location: [../training](../training) (PyTorch, eager). Exporter writes numpy artifacts to `model/best` and `training/best_model` via `training/export.py` when loss improves.
- Smoke commands: `python -m training.train --synthetic --steps 3 --amp 0 --checkpoint_dir /tmp/seg_ckpt` or `python -m training.train --config configs/datasets/naip_3dep_example.yaml --steps 1 --amp 0 --checkpoint_dir /tmp/seg_ckpt`.

## Editing & conventions
- Keep UI thread lean; any heavy IO/inference must stay in `QgsTask` workers.
- When adding status text, include numeric hints (`42%`, `3/10`) so the parser advances progress.
- New runtime outputs should go through `render_raster()` to preserve CRS/extents; avoid writing directly to disk elsewhere.
