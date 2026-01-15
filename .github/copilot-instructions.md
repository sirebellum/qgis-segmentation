# QGIS Segmenter – Copilot Instructions

## Architecture
- Plugin shell: [../segmenter.py](../segmenter.py) owns the QGIS-facing `Segmenter` class, wires dialog actions to `predict()`/`stop_current_task()`, and dispatches heavy work onto `QgsTask` via `run_task()`. Keep UI thread light.
- Core engine: [../funcs.py](../funcs.py) contains `_materialize_raster()`, `predict_kmeans()`, `predict_cnn()`, latent-KNN refinement, blur utilities, and cancellation hooks. Follow the existing status callbacks + `CancellationToken` checks inside loops.
- Rendering: [../qgis_funcs.py](../qgis_funcs.py) writes temporary GeoTIFFs and re-adds them with the source extent/CRS. Route new raster outputs through `render_raster()`.
- UI: [../segmenter_dialog.py](../segmenter_dialog.py) and [../segmenter_dialog_base.ui](../segmenter_dialog_base.ui). If the `.ui` changes, re-run `pyuic5` and keep widget names that `Segmenter` expects.

## Models & data
- TorchScript CNNs live under [../models](../models) as `model_<tile>.pth` and must return `(mask, features)`; `Segmenter.load_model()` wraps them. Update [../func_test.py](../func_test.py) mocks if the tuple contract changes.
- Next-gen numpy runtime reads [../model/best](../model/README.md) (`model.npz`, `meta.json`) when the "Next-Gen (Numpy)" option is chosen. Exported automatically from `python -m training.train` unless `--no-export`.

## Dependencies
- `ensure_dependencies()` in [../dependency_manager.py](../dependency_manager.py) runs at import; any top-level torch/numpy/sklearn import triggers vendor installs. Keep `_package_specs()` and [../README.md](../README.md) in sync.
- Environment toggles: `SEGMENTER_SKIP_AUTO_INSTALL`, `SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`, `SEGMENTER_PYTHON`, `SEGMENTER_SKIP_PROFILING`.

## Runtime patterns
- Progress/logging: emit human-readable strings; `Segmenter.log_status()` is UI-safe, `_emit_status()` in [../funcs.py](../funcs.py) is for worker threads, and `_maybe_update_progress_from_message()` parses percentages/steps to advance the bar.
- Cancellation: call `_maybe_raise_cancel()` inside tile/chunk loops (CNN tiling, latent KNN, blur) so `Task.cancel()` is responsive.
- Heuristics: `Segmenter._collect_heuristics()` → `_build_heuristic_overrides()` (tile size + latent KNN knobs) → `legacy_cnn_segmentation()`. Blur tuning lives in `_legacy_blur_config()` → `blur_segmentation_map()`.

## Performance & profiling
- Adaptive batching/prefetch live in [../perf_tuner.py](../perf_tuner.py); `load_or_profile_settings()` caches `perf_profile.json` keyed by device. Respect `SEGMENTER_SKIP_PROFILING` and surface summaries via `log_status`.
- `predict_cnn()` uses `_recommended_batch_size()` + `_prefetch_batches()`; keep these consistent with profiling outputs.

## Developer workflow
- Tests: `pytest func_test.py` for runtime/heuristics/regressions; use markers (`-k predict_cnn`, `-m performance`) for focus.
- Qt assets: `pyrcc5 resources.qrc -o resources.py` after icon/resource edits; `pb_tool deploy` (see [../pb_tool.cfg](../pb_tool.cfg)) to sync into a QGIS profile.
- Docs/metadata: update [../README.md](../README.md) and [../metadata.txt](../metadata.txt) when changing runtime behavior, dependencies, or user-facing UX.
- Training smoke: synthetic run `python -m training.train --synthetic --steps 3 --amp 0 --checkpoint_dir /tmp/seg_ckpt`; manifest-backed example `python -m training.train --config configs/datasets/naip_3dep_example.yaml --steps 1 --amp 0 --checkpoint_dir /tmp/seg_ckpt`.
