<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Model Notes

## Plugin Inference Pipeline

### Inputs
- 3-band GDAL raster layer validated in `segmenter._is_supported_raster_layer`
- Must be RGB GeoTIFF (`.tif/.tiff`), provider `gdal`, band count 3
- Data materialized via `runtime/io._materialize_raster` (GDAL loader)

### Runtime Paths
User selects **CNN** (TorchScript) or **K-Means** in the UI:

**CNN Path** (`execute_cnn_segmentation`):
1. Load TorchScript model from `models/model_{resolution}.pth`
2. Fit global centers once using `fit_global_cnn_centers`
3. Tile input (default 512px, configurable 192–768px)
4. Stream assignment per chunk via `predict_cnn_with_centers`
5. Aggregate chunk scores, apply optional post-smoothing
6. Stitch and return labels

**K-Means Path** (`execute_kmeans_segmentation`):
1. Smooth and pool descriptors at resolution
2. Fit global centers once using torch K-Means
3. Stream assignment per chunk
4. Upsample to original resolution
5. Return labels

Both paths use **global centers** to ensure consistent label IDs across chunks (no per-chunk relabeling).

### Output
- Segmentation labels as uint8 numpy array (class IDs start at 0)
- Rendered as GeoTIFF via `qgis_funcs.render_raster` preserving extent/CRS
- Layer opacity set to 1.0

## Controls and Heuristics

### User Inputs
- **Segment count**: number of output classes (2–16 typical)
- **Resolution**: low/medium/high → model_16/8/4.pth
- **Sliders**: smoothness, speed, accuracy

### Slider Effects
- **Smoothness**: blur kernel size (1–9px), iterations (1–2)
- **Speed**: CNN tile size (192–768px), sampling scale
- **Accuracy**: sampling scale, latent KNN overrides

### Latent KNN (optional)
Configurable via heuristic overrides:
- `mix`: soft assignment mixing weight
- `temperature`: softmax temperature
- `neighbors`: K for KNN
- `iterations`: refinement passes

## Dependencies
- **torch**: required, vendor-installed via `dependency_manager.py`
- **NumPy**: required, vendor-installed
- **GDAL**: required for raster IO and rendering (system install)
- **scikit-learn**: NOT required (removed; K-Means is torch-only)

### Environment Overrides
- `SEGMENTER_SKIP_AUTO_INSTALL`: skip vendor install
- `SEGMENTER_PYTHON`: interpreter override
- `SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`: torch wheel selection

## Training (Isolated)
Training produces artifacts not yet consumed by the plugin:
- Student embeddings via distillation
- Autoencoder reconstruction loss (training-only decoder)
- Exports to numpy artifacts (`model.npz`, `meta.json`, `metrics.json`)

See [training/README.md](../../training/README.md) for training pipeline details.
