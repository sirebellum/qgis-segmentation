# QGIS Segmenter – Copilot Instructions

## Architecture Snapshot
- [../segmenter.py](../segmenter.py) hosts the plugin entry point: the `Segmenter` class wires into QGIS, spins up `SegmenterDialog`, and schedules heavy work through `QgsTask` via `run_task()`.
- Segmentation happens in [../funcs.py](../funcs.py) through `predict_kmeans()`, `predict_cnn()`, and `tile_raster()`; they expect numpy channel-first rasters and emit status strings through optional callbacks.
- Rendering is decoupled: [../qgis_funcs.py](../qgis_funcs.py) turns numpy arrays back into temporary GeoTIFFs and adds them as layers with the source extent/CRS.
- UI chrome lives in [../segmenter_dialog.py](../segmenter_dialog.py) + [../segmenter_dialog_base.ui](../segmenter_dialog_base.ui); tweak the `.ui`, then re-run PyQt codegen if you add widgets.
- Pretrained TorchScript models ship in [../models](../models); model filenames encode tile resolution (`model_4.pth`, etc.).

## Dependency Model
- `ensure_dependencies()` in [../dependency_manager.py](../dependency_manager.py) runs at import time (both in `segmenter.py` and `funcs.py`) so any module-level import will try to bootstrap torch/numpy/sklearn into `vendor/`.
- Respect env knobs advertised in [../README.md](../README.md): `SEGMENTER_SKIP_AUTO_INSTALL`, `SEGMENTER_TORCH_INDEX_URL`, `SEGMENTER_TORCH_SPEC`, and `SEGMENTER_PYTHON`; altering dependency logic without updating docs will surprise operators.
- When adding new third-party libs, extend `_package_specs()` so offline installers and the bootstrap dialog stay in sync.

## Runtime Patterns
- Long operations must emit progress via `status_callback` to keep the dialog responsive; reuse `_emit_status()` in [../funcs.py](../funcs.py) or `Segmenter.log_status()` as appropriate.
- `Task.finished()` is responsible for raster rendering; if you change prediction outputs, update the kwargs contract passed in `Segmenter.predict()`.
- CUDA vs CPU selection happens once in `Segmenter.run()`; new GPU-dependent code should branch on `self.device` the same way and fall back to CPU.
- Tiling math assumes rasters are RGB with shape `(3, H, W)`; validate or coerce upstream if you start supporting other band counts.

## Developer Workflow
- Local unit coverage focuses on numerical helpers; run `python -m unittest func_test` from the repo root to exercise tiling/K-Means/CNN plumbing without QGIS.
- Keep `requirements.txt` aligned with what `dependency_manager` installs so editors and CI get the same transitive set.
- Use [../pb_tool.cfg](../pb_tool.cfg) when syncing into a QGIS profile (`pb_tool deploy` targets the QGIS plugin dir); remember to rebuild resources.qrc if you add icons (`pyrcc5 resources.qrc -o resources.py`).
- The release-ready copy under [../zip_build/segmenter](../zip_build/segmenter) mirrors the live plugin; update it after making source changes or automate the copy in your build.

## Feature Tips
- K-Means uses block sampling (max 10k tiles) before clustering; if you tune performance, preserve the sampling step to avoid memory spikes.
- CNN inference expects `torch.jit.load()` outputs returning `(mask, features)`; mocks in [../func_test.py](../func_test.py) mirror this contract and should be adjusted if the model interface changes.
- Status text is buffered before the dialog opens (`_status_buffer`); if you log from background threads before UI init, call `_flush_status_buffer()` once widgets exist.

## Documentation & Support
- User-facing behavior, dependency guidance, and screenshots live in [../README.md](../README.md); keep that file updated when workflow or UX changes so support links don't drift.
- Licensing, metadata, and packaging data are sourced from [../metadata.txt](../metadata.txt) and [../LICENSE](../LICENSE); remember to update both if you rebrand or relicense.
