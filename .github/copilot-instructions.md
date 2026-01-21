QGIS Segmenter – Copilot Instructions

## Architecture & flow
- UI entry: [segmenter.py](../segmenter.py) `Segmenter` builds the dialog from [segmenter_dialog_base.ui](../segmenter_dialog_base.ui) via [segmenter_dialog.py](../segmenter_dialog.py); buttons call `predict()`/`stop_current_task()`.
- Task dispatch: `predict()` validates GDAL `.tif/.tiff` 3-band rasters and segment count, then enqueues a `QgsTask` via `run_task()` to keep the UI thread idle.
- Runtime path: legacy flow calls `legacy_kmeans_segmentation`/`legacy_cnn_segmentation` in [funcs.py](../funcs.py); CNN path tiles/stitches and may run latent KNN refinement before optional blur.
- Rendering: [qgis_funcs.py](../qgis_funcs.py) `render_raster()` writes a temp GeoTIFF with source extent/CRS and re-adds it as `QgsRasterLayer` (opacity 1.0).

## Dependencies & env toggles
- Auto-install: [dependency_manager.py](../dependency_manager.py) `ensure_dependencies()` vendors NumPy/Torch/sklearn into `vendor/` unless `SEGMENTER_SKIP_AUTO_INSTALL=1`; uses CPU torch on macOS by default.
- Overrides: `SEGMENTER_PYTHON`/`SEGMENTER_PIP_EXECUTABLE` to pick the interpreter/pip; `SEGMENTER_TORCH_SPEC`/`SEGMENTER_TORCH_INDEX_URL` to pin wheels; `SEGMENTER_RUNTIME_BACKEND` (`auto`/`torch`/`numpy`) and `SEGMENTER_DEVICE` (`auto`/`cuda`/`mps`/`cpu`) steer runtime.
- GPU policy: CUDA uses ~0.9% free VRAM, MPS ~0.75%, CPU uses system RAM; tiling/batching respects these budgets.

## Models & data contract
- Artifacts: runtime expects channel-first RGB uint8/float32 and returns uint8 labels. Legacy TorchScript models live under [models/](../models) (`model_4/8/16.pth`).
- Input enforcement: `_is_supported_raster_layer()` (3 bands, GDAL provider, .tif/.tiff) and `_materialize_raster()` enforce format; errors are surfaced via UI logs.
- Tiling/stitching: `tile_raster()` pads to `TILE_SIZE` (512 legacy default); `_ChunkAggregator` smooths chunk edges and harmonizes label palettes.

## UI/status patterns
- Progress: `_maybe_update_progress_from_message()` parses worker messages to drive `jobProgress`; stage bounds in `PROGRESS_STAGE_MAP` enforce monotonic updates.
- Cancellation: `CancellationToken` bound to `QgsTask`; call `_maybe_raise_cancel()` inside loops (already in kmeans/CNN flows) and let `Task.cancel()` propagate.
- Logging: use `Segmenter.log_status()` for UI-safe messages; worker threads should use the `status_callback` passed into segmentation functions.

## Testing & workflows
- Runtime/tests: `pytest` (root) covers tiling, runtime selection, plugin import, numpy/torch smoke; training tests live in [training/tests](../training/tests) (`pytest training/tests`).
- Packaging: `pb_tool zip` builds the distributable; `pb_tool deploy` (see [pb_tool.cfg](../pb_tool.cfg)) syncs into a QGIS profile.
- Qt assets: after editing `resources.qrc` run `pyrcc5 resources.qrc -o resources.py`; if `segmenter_dialog_base.ui` changes, re-run `pyuic5 segmenter_dialog_base.ui -o segmenter_dialog.py` and keep widget names stable.

## Training (isolated from plugin runtime)
- Scope: eager PyTorch unsupervised pipeline in [training](../training); see [training/README.md](../training/README.md) for contract and commands.
- Smoke: `python -m training.train --synthetic --steps 3 --amp 0 --checkpoint_dir /tmp/seg_ckpt --seed 123` (no TorchScript export required in this phase).

## Dataset tooling
- Headers: schema/workflow in [docs/dataset/HEADERS.md](../docs/dataset/HEADERS.md); generate with `python -m training.datasets.generate_headers --dry-run` (defaults to `training/datasets/extracted`), then build shards with `python -m training.datasets.build_shards --overwrite --seed 123 --shard-size 512 --threads 4`.
- Ingestion stub: [scripts/datasets_ingest](../scripts/datasets_ingest/README.md) is offline-only; use `python -m scripts.datasets_ingest.cli --list-providers` or `--provider placeholder --dataset demo --sample-size 2` to plan manifests (no downloads/GDAL).

## Editing & conventions
- Keep the QGIS UI thread lean; heavy IO/inference stays inside `QgsTask` workers.
- Status strings should include numeric hints (`42%`, `3/10`) so progress parsing advances.
- Route new runtime outputs through `render_raster()` to preserve CRS/extents; avoid ad-hoc disk writes elsewhere.
