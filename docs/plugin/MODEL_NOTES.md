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

### Runtime Path
The plugin uses K-Means segmentation exclusively:

**K-Means Path** (`execute_kmeans_segmentation`):
1. Smooth and pool descriptors at resolution
2. Fit global centers once using torch K-Means
3. Stream assignment per chunk using global centers
4. Upsample to original resolution
5. Apply optional post-smoothing (when checkbox enabled)
6. Return labels

The K-Means path uses **global centers** to ensure consistent label IDs across chunks (no per-chunk relabeling).

### Output
- Segmentation labels as uint8 numpy array (class IDs start at 0)
- Rendered as GeoTIFF via `qgis_funcs.render_raster` preserving extent/CRS
- Layer opacity set to 1.0

## Controls and Heuristics

### User Inputs
- **Segment count**: number of output classes (2–16 typical)
- **Resolution**: low/medium/high → block sizes 16/8/4
- **Smoothing toggle**: checkbox to enable/disable post-process smoothing (default: disabled)
- **Sliders**: smoothness, speed, accuracy — 3 discrete levels (Low/Medium/High)

### Slider Effects
- **Smoothness**: blur kernel size (Low=1px, Medium=3px, High=7px), iterations (1–2); only applies when smoothing checkbox is enabled
- **Speed**: chunk sizes, block resolution scaling (Low=conservative/slower, High=aggressive/faster)
- **Accuracy**: sampling density for K-Means center fitting (Low=fast/coarse, High=slow/precise)

## Dependencies
- **torch**: required, vendor-installed via `dependency_manager.py`
- **NumPy**: required, vendor-installed
- **GDAL**: required for raster IO and rendering (system install)
- **scikit-learn**: NOT required (K-Means is torch-only)

### Environment Overrides
- `SEGMENTER_SKIP_AUTO_INSTALL`: skip vendor install
- `SEGMENTER_PYTHON`: interpreter override
- `SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`: torch wheel selection

## Deprecated / Removed

### CNN Runtime (removed as of Phase 2)
The CNN inference path has been removed:
- No model dropdown in UI
- No TorchScript model files required
- `models/` directory and `runtime/cnn.py` are not used by the runtime

### Training Pipeline (not shipped)
Training code is isolated from the plugin runtime and is not part of the distributed plugin.
