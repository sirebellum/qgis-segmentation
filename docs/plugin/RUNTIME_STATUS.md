<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Runtime Status (QGIS Plugin)

- **Path**: QGIS UI → `Segmenter.predict()` → `QgsTask` → `funcs.predict_nextgen_numpy` → runtime selector (`model/runtime_backend.py`) chooses torch (CUDA/MPS/CPU) or numpy → `qgis_funcs.render_raster`.
- **Dependencies**: `dependency_manager.ensure_dependencies()` now runs from `classFactory` to vendor-install NumPy into `vendor/` unless `SEGMENTER_SKIP_AUTO_INSTALL` is set. Torch is optional: set `SEGMENTER_ENABLE_TORCH=1` to attempt install, or pre-install a GPU-capable build; the selector falls back to numpy if torch is unavailable.
- **Inputs (hard requirement)**: GDAL raster layer, provider `gdal`, source `.tif/.tiff`, exactly 3 bands. Enforced in `segmenter._is_supported_raster_layer` and `_materialize_raster` (raises with band count/extension in message).
- **Model artifacts**: Shipped inside the plugin under `model/best` (`meta.json`, `model.npz`). `pb_tool.cfg` `extra_dirs` includes `model/`; ship the same directory as a GitHub artifact (use Git LFS if size exceeds repo limits).
- **Runtime behavior**: Tiling/stitching remains numpy-based; `predict_nextgen_numpy` calls whichever backend exposes `predict_labels`. Cancellation via `CancellationToken`; progress parsed from worker status strings. Torch path prefers GPU devices and logs fallback to numpy on import/device errors.
