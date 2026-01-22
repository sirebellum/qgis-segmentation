<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Runtime Status (QGIS Plugin)

## Current Path
```
QGIS UI → Segmenter.predict() → QgsTask → runtime/pipeline.py → qgis_funcs.render_raster
```

## Dependencies
- `dependency_manager.ensure_dependencies()` runs on import
- Vendor-installs torch and NumPy into `vendor/` unless `SEGMENTER_SKIP_AUTO_INSTALL`
- Torch index/spec configurable via env vars
- Device preference: CUDA → MPS → CPU (set in `segmenter.run()`)

## Input Requirements (Hard)
- GDAL raster layer
- Provider: `gdal`
- Source: `.tif` or `.tiff`
- Bands: exactly 3 (RGB)
- Enforced in `segmenter._is_supported_raster_layer`

## Model Artifacts
- **Active**: TorchScript CNN weights in `models/` (`model_4.pth`, `model_8.pth`, `model_16.pth`)
- **K-Means**: torch-only clustering backend (no scikit-learn)

## Runtime Behavior

### K-Means (`execute_kmeans_segmentation`)
1. Smooth and pool descriptors at resolution
2. Fit global centers once via torch K-Means
3. Stream assignment per chunk
4. Upsample labels to original resolution

### CNN (`execute_cnn_segmentation`)
1. Fit global centers once using `fit_global_cnn_centers`
2. Tile input (default 512px, slider-tuned 192–768px)
3. Stream assignment per chunk via `predict_cnn_with_centers`
4. Aggregate chunk scores
5. Apply optional post-smoothing
6. Stitch and return labels

### Key Invariants
- **Global centers**: fitted once per run (no per-chunk relabeling)
- **Streaming assignment**: labels consistent across chunk boundaries
- **Cancellation**: `CancellationToken` checked throughout

### Optional Features
- **Post-smoothing**: blur kernel/iterations on outputs
- **Latent KNN**: soft refinement (disabled by default)
- **Texture autoencoder**: remap (disabled by default)

## Environment Variables
| Variable | Purpose |
|----------|---------|
| `SEGMENTER_SKIP_AUTO_INSTALL` | Skip vendor dependency install |
| `SEGMENTER_PYTHON` | Python interpreter override |
| `SEGMENTER_PIP_EXECUTABLE` | pip executable override |
| `SEGMENTER_TORCH_SPEC` | Torch version spec (e.g., `torch>=2.3`) |
| `SEGMENTER_TORCH_INDEX_URL` | Torch wheel index URL |

## Removed / Non-Existent
The following are NOT present in the current codebase:
- `model/` directory (runtime_numpy.py, runtime_backend.py, runtime_torch.py)
- `scripts/datasets_ingest/` (ingestion scaffold)
- scikit-learn dependency (K-Means is torch-only)

## Tests (91 passing)
```bash
.venv/bin/python -m pytest tests/ -q
```
Key test files:
- **Core**: test_funcs_runtime.py (21 tests)
- **Routing**: test_runtime_pipeline_routing.py, test_runtime_no_legacy_usage.py
- **K-Means**: test_kmeans_backend_routing.py, test_predict_kmeans_*.py
- **Chunks**: test_global_centers_*.py, test_label_consistency_across_chunks.py
- **Runtime modules**: test_runtime_adaptive.py, test_runtime_common.py, test_runtime_io.py, test_runtime_latent.py, test_runtime_chunking.py

## Output
- uint8 label map rendered as GeoTIFF
- Preserves source extent/CRS
- Opacity 1.0
- Layer name: `{input}_{model}_{segments}_{resolution}`
