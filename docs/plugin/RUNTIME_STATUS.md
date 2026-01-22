<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Runtime Status (QGIS Plugin)

- **Path (current runtime)**: QGIS UI → `Segmenter.predict()` → `QgsTask` → `funcs.execute_cnn_segmentation` **or** `funcs.execute_kmeans_segmentation` (facade into `runtime/`) → `qgis_funcs.render_raster`.
- **Dependencies**: `dependency_manager.ensure_dependencies()` runs on import to vendor-install torch and NumPy unless `SEGMENTER_SKIP_AUTO_INSTALL` is set. Torch index/spec can be overridden (`SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`). Device preference in `segmenter.py` is CUDA → MPS → CPU.
- **Inputs (hard requirement)**: GDAL raster layer, provider `gdal`, source `.tif/.tiff`, exactly 3 bands. Enforced in `segmenter._is_supported_raster_layer`; file loading via `_materialize_raster` uses GDAL.
- **Model artifacts (active)**: TorchScript CNN weights under `models/` (`model_4.pth`, `model_8.pth`, `model_16.pth`). K-Means uses the torch-only clustering backend. **Next-gen numpy artifacts in `model/best` exist but are not consumed; runtime update is deferred until training completes.**
- **Runtime behavior**: CNN path tiles (default 512px, slider-tuned 192–768px), fits **global centers once**, streams assignment per chunk, aggregates chunk scores, applies optional post-smoothing, and stitches. K-Means likewise fits global centers once and streams assignment before upsampling. No per-chunk relabeling is allowed. Cancellation via `CancellationToken`; progress parsed from worker status strings; texture remap is available but disabled by default in the plugin.
