# QGIS Segmenter – Copilot Instructions

## Architecture
- Plugin shell: [../segmenter.py](../segmenter.py) owns the QGIS-facing `Segmenter` class, wires dialog actions to `predict()`/`stop_current_task()`, and dispatches heavy work onto `QgsTask` via `run_task()`. Keep UI thread light.
- Core engine: [../funcs.py](../funcs.py) contains `_materialize_raster()`, tiling/stitching helpers, and the numpy-only `predict_nextgen_numpy()` path. Follow the existing status callbacks + `CancellationToken` checks inside loops.
- Rendering: [../qgis_funcs.py](../qgis_funcs.py) writes temporary GeoTIFFs and re-adds them with the source extent/CRS. Route new raster outputs through `render_raster()`.
- UI: [../segmenter_dialog.py](../segmenter_dialog.py) and [../segmenter_dialog_base.ui](../segmenter_dialog_base.ui). If the `.ui` changes, re-run `pyuic5` and keep widget names that `Segmenter` expects.

## Models & data
- Runtime is numpy-only: [../model/best](../model/README.md) provides `model.npz` + `meta.json` consumed by `load_runtime_model()` in [../segmenter.py](../segmenter.py). Exported automatically from `python -m training.train` unless `--no-export`.

## Dependencies
- `ensure_dependencies()` in [../dependency_manager.py](../dependency_manager.py) runs at plugin initialization; keep `_package_specs()` and [../README.md](../README.md) in sync.
- Environment toggles: `SEGMENTER_SKIP_AUTO_INSTALL`, `SEGMENTER_PYTHON`, `SEGMENTER_SKIP_PROFILING`.

## Runtime patterns
- Progress/logging: emit human-readable strings; `Segmenter.log_status()` is UI-safe, `_emit_status()` in [../funcs.py](../funcs.py) is for worker threads, and `_maybe_update_progress_from_message()` parses percentages/steps to advance the bar.
- Cancellation: call `_maybe_raise_cancel()` inside tile/chunk loops so `Task.cancel()` is responsive.

## Performance & profiling
- `perf_tuner.py` is a shim; runtime does not profile or prefetch.

## Developer workflow
- Tests: `pytest func_test.py` for runtime/heuristics/regressions; use markers (`-k predict_cnn`, `-m performance`) for focus.
- Qt assets: `pyrcc5 resources.qrc -o resources.py` after icon/resource edits; `pb_tool deploy` (see [../pb_tool.cfg](../pb_tool.cfg)) to sync into a QGIS profile.
- Docs/metadata: update [../README.md](../README.md) and [../metadata.txt](../metadata.txt) when changing runtime behavior, dependencies, or user-facing UX.
- Training smoke: synthetic run `python -m training.train --synthetic --steps 3 --amp 0 --checkpoint_dir /tmp/seg_ckpt`; manifest-backed example `python -m training.train --config configs/datasets/naip_3dep_example.yaml --steps 1 --amp 0 --checkpoint_dir /tmp/seg_ckpt`.
