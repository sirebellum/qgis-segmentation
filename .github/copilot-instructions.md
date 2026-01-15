# QGIS Segmenter – Copilot Instructions

## Core Layout
- [../segmenter.py](../segmenter.py) instantiates `SegmenterDialog`, wires UI buttons to `predict()`/`stop_current_task()`, and pushes heavy work onto `QgsTask` via `run_task()`. Keep GUI code responsive; anything CPU-bound should live outside the main thread.
- [../funcs.py](../funcs.py) is the numerical engine: `_materialize_raster()` ensures channel-first arrays, `predict_kmeans()` handles block-sampled clustering, and `predict_cnn()` manages tiling, batching, latent-KNN refinement, and optional blur. Mimic the existing status callbacks + cancellation tokens when extending these paths.
- Rendering is isolated in [../qgis_funcs.py](../qgis_funcs.py): it converts numpy labels to temporary GeoTIFFs and adds them to the current project with the source extent/CRS. New raster-like outputs should flow through `render_raster()` so metadata stays consistent.
- UI shell lives in [../segmenter_dialog.py](../segmenter_dialog.py) + [../segmenter_dialog_base.ui](../segmenter_dialog_base.ui). After changing the `.ui`, re-run `pyuic5` if you need generated Python, and remember to keep widget names aligned with what `Segmenter` expects.

## Dependencies & Models
- `ensure_dependencies()` in [../dependency_manager.py](../dependency_manager.py) executes at import time; any top-level import of torch/numpy/sklearn will trigger vendor installs. Update `_package_specs()` plus [../README.md](../README.md) when adding libraries, and honor env toggles (`SEGMENTER_SKIP_AUTO_INSTALL`, `SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`, `SEGMENTER_PYTHON`).
- Pretrained TorchScript files ship in [../models](../models); the numeric suffix indicates tile resolution (`model_4.pth`, `model_8.pth`, etc.). `Segmenter.load_model()` expects `.pth` files that return `(mask, features)`; adjust mocks in [../func_test.py](../func_test.py) if you change that contract.

## Runtime Patterns
- All long-running work must emit progress strings. Use `Segmenter.log_status()` for UI-safe logs and `_emit_status()` in [../funcs.py](../funcs.py) for worker updates so `_maybe_update_progress_from_message()` can keep the progress bar moving.
- Cancellation flows through `CancellationToken`; call `_maybe_raise_cancel()` anywhere that loops over tiles/chunks so `Task.cancel()` works instantly.
- CNN heuristics are centralized: `Segmenter._collect_heuristics()` maps sliders to speed/accuracy, `_build_heuristic_overrides()` feeds tile sizes + latent KNN settings into `legacy_cnn_segmentation()`, and `_legacy_blur_config()` tunes the optional blur. When adding knobs, thread them through this pipeline rather than hard-coding per-model tweaks.

## Performance Tooling
- [../perf_tuner.py](../perf_tuner.py) (`load_or_profile_settings()`) profiles GPU throughput once and caches results in `perf_profile.json`. Honor `SEGMENTER_SKIP_PROFILING` before launching the thread, and surface user-facing summaries via `Segmenter.log_status()`.
- Memory-aware tiling relies on `_derive_chunk_size()` and `_ChunkAggregator` helpers; if you alter tiling math, update `recommended_chunk_plan()` and the associated tests in [../func_test.py](../func_test.py).

## Developer Workflow
- Run tests with `pytest func_test.py` (the suite relies on pytest fixtures/marks). For focused runs, use markers like `pytest func_test.py -k predict_cnn`.
- Regenerate Qt resources with `pyrcc5 resources.qrc -o resources.py` when icons change, and use `pb_tool deploy` (see [../pb_tool.cfg](../pb_tool.cfg)) to sync into a QGIS profile.
- Keep `requirements.txt` aligned with `dependency_manager.py` so editors/CI match the in-app bootstrapper. After changing runtime behavior, update [../README.md](../README.md) plus [../metadata.txt](../metadata.txt) to keep end-user docs in sync.
