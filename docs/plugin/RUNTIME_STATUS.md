<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Runtime Status (NumPy-only plugin path)

Purpose: snapshot of the current QGIS plugin runtime as implemented in this repo. Scope is runtime code only; no training or dataset changes.

## Module map
- UI + dispatch: [segmenter.py](../../segmenter.py) owns plugin wiring, dialog lifecycle, validation, progress/log routing, and `QgsTask` submission.
- Dialog shell: [segmenter_dialog.py](../../segmenter_dialog.py) and [segmenter_dialog_base.ui](../../segmenter_dialog_base.ui) define the minimal UI (layer picker, segment count, log/progress, feedback/support buttons).
- Engine: [funcs.py](../../funcs.py) handles raster materialization, tiling/stitching, cancellation/status helpers, and `predict_nextgen_numpy`.
- Rendering: [qgis_funcs.py](../../qgis_funcs.py) writes a temporary GeoTIFF and re-adds it to the project with extent/CRS preserved.
- Model runtime: [model/runtime_numpy.py](../../model/runtime_numpy.py) loads `model/best` artifacts (`meta.json`, `model.npz`) and exposes `predict_labels`.
- Bootstrap/shims: [dependency_manager.py](../../dependency_manager.py) installs NumPy into `vendor/` on import; [perf_tuner.py](../../perf_tuner.py) is a no-op profiling shim.

## Entrypoints and call flow
- User clicks the plugin action → [Segmenter.run](../../segmenter.py) shows the dialog and seeds defaults.
- Predict button → [Segmenter.predict](../../segmenter.py) validates a 3-band GDAL raster and a positive integer segment count, then enqueues `predict_nextgen_numpy` via `run_task` (QgsTask).
- Task thread executes [predict_nextgen_numpy](../../funcs.py): materialize raster → tile/stitch loop with cancellation checks → per-tile `model.predict_labels` → assemble canvas and trim padding.
- On task success, [Task.finished](../../segmenter.py) calls [render_raster](../../qgis_funcs.py) to emit a temp GeoTIFF and add it as a raster layer. Progress is updated throughout via status messages parsed in the UI thread.

## Data flow
- Inputs: selected QGIS raster layer; validated to GDAL provider, 3 bands, `.tif/.tiff` extension. Path handed to worker; IO occurs off the UI thread.
- Processing: tile size fixed at 256; tiles cast to float32 before model inference; padding added to cover full tiles and trimmed after stitching.
- Outputs: uint8 label map shaped like the source raster; stored as GeoTIFF in tempdir; opacity forced to 1.0 when added to the project.

## Model runtime contract
- Artifacts: `model/best/meta.json` + `model/best/model.npz` (required). Loader raises `FileNotFoundError` if missing.
- Inputs to `predict_labels`: RGB array `[3,H,W]` float32; runtime applies `input_scale` then per-channel `(x - mean) / std` from metadata.
- Behavior: stride-4 encoder + soft k-means head; `k` capped to `[2, max_k]`; optional box-filter smoothing from metadata defaults; outputs uint8 labels `[H,W]`.

## Configuration and knobs
- User-facing: segment count (integer > 0). Tile size is constant (256). No heuristic/blur sliders in UI.
- Env toggles: `SEGMENTER_SKIP_AUTO_INSTALL`, `SEGMENTER_PYTHON`, `SEGMENTER_TORCH_SPEC/INDEX_URL` are ignored by runtime (Torch unused) but respected by `dependency_manager` for NumPy bootstrap.

## Error handling, logging, cancellation
- Validation errors reset the progress bar and log to the dialog history.
- Worker errors set task to failed and update UI progress state to "Failed"; success sets "Segmentation complete".
- Cancellation: `CancellationToken` wired to `QgsTask.cancel`; checked before/inside tile loop; UI shows "Cancelling task..." and disables Stop button.
- Progress: status strings parsed for percent/step tokens to stage-aware progress updates; worker emits periodic "Next-gen numpy inference X%".

## Known gaps / risks
- `funcs.py` retains unused torch/prefetch helpers that are not referenced by the plugin path; ensure they stay unused on the runtime path.
- No explicit nodata handling beyond padding trim; input must be a valid 3-band GDAL raster.
- If `model/best` is absent, prediction aborts with a user-facing log message and no task dispatch.

## Snapshot limits
- Based on the current workspace contents; no external network or training artifacts were inspected.
