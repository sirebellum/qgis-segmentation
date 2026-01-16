<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Runtime Status (QGIS Plugin)

- **Path**: QGIS UI → `Segmenter.predict()` → `QgsTask` → `funcs.predict_nextgen_numpy` → `model/runtime_numpy.py` → `qgis_funcs.render_raster`.
- **Dependencies**: `dependency_manager.ensure_dependencies()` now runs from `classFactory` to vendor-install NumPy into `vendor/` unless `SEGMENTER_SKIP_AUTO_INSTALL` is set. No torch/scikit-learn on the runtime path.
- **Inputs (hard requirement)**: GDAL raster layer, provider `gdal`, source `.tif/.tiff`, exactly 3 bands. Enforced in `segmenter._is_supported_raster_layer` and `_materialize_raster` (raises with band count/extension in message).
- **Model artifacts**: Shipped inside the plugin under `model/best` (`meta.json`, `model.npz`). `pb_tool.cfg` `extra_dirs` includes `model/`; ship the same directory as a GitHub artifact (use Git LFS if size exceeds repo limits).
- **Runtime behavior**: Numpy-only tiling/stitching; cancellation via `CancellationToken`; progress parsed from worker status strings. Torch/prefetch helpers removed from runtime modules.
