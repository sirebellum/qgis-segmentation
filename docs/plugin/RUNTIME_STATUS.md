<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Runtime Status (QGIS Plugin)

- **Path (current legacy runtime)**: QGIS UI → `Segmenter.predict()` → `QgsTask` → `funcs.legacy_cnn_segmentation` **or** `funcs.legacy_kmeans_segmentation` → `qgis_funcs.render_raster`.
- **Dependencies**: `dependency_manager.ensure_dependencies()` runs on import to vendor-install torch, NumPy, and scikit-learn unless `SEGMENTER_SKIP_AUTO_INSTALL` is set. Torch index/spec can be overridden (`SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`). Device preference in `segmenter.py` is CUDA → MPS → CPU; historical runtime selector env flags are inactive in the restored legacy flow.
- **Inputs (hard requirement)**: GDAL raster layer, provider `gdal`, source `.tif/.tiff`, exactly 3 bands. Enforced in `segmenter._is_supported_raster_layer`; file loading via `_materialize_raster` uses GDAL.
- **Model artifacts (active)**: Legacy TorchScript CNN weights under `models/` (`model_4.pth`, `model_8.pth`, `model_16.pth`). K-Means uses scikit-learn only. **Next-gen numpy artifacts in `model/best` exist but are not consumed; runtime update is deferred until training completes.**
- **Runtime behavior**: CNN path tiles (default 512px, slider-tuned 192–768px), runs inference on the selected device, applies optional legacy blur, and stitches; K-Means clusters block samples and upsamples. Cancellation via `CancellationToken`; progress parsed from worker status strings; blur smoothing is optional for both paths.
