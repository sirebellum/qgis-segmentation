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

## Segmentation Backend
- **K-Means**: torch-only clustering backend (no scikit-learn)
- **Global centers**: fitted once per run for consistent labels across chunks

## Runtime Behavior

### K-Means (`execute_kmeans_segmentation`)
1. Smooth and pool descriptors at resolution
2. Fit global centers once via torch K-Means
3. Stream assignment per chunk using global centers
4. Upsample labels to original resolution
5. Apply optional post-smoothing (when checkbox enabled)

### Key Invariants
- **Global centers**: fitted once per run (no per-chunk relabeling)
- **Streaming assignment**: labels consistent across chunk boundaries
- **Cancellation**: `CancellationToken` checked throughout

### Optional Features
- **Post-smoothing**: opt-in via UI checkbox (default: disabled); blur kernel/iterations controlled by smoothness slider

### UI Controls
- **Smoothness slider**: 3 discrete levels (Low/Medium/High) with info icon; controls blur kernel (1/3/7px) and iterations; only applies when smoothing checkbox is checked

## Environment Variables
| Variable | Purpose |
|----------|---------|
| `SEGMENTER_SKIP_AUTO_INSTALL` | Skip vendor dependency install |
| `SEGMENTER_PYTHON` | Python interpreter override |
| `SEGMENTER_PIP_EXECUTABLE` | pip executable override |
| `SEGMENTER_TORCH_SPEC` | Torch version spec (e.g., `torch>=2.3`) |
| `SEGMENTER_TORCH_INDEX_URL` | Torch wheel index URL |

## Removed / Not Used by Runtime
The following are NOT used by the runtime segmentation path:
- `model/` directory (legacy runtime selectors)
- `models/` directory (TorchScript CNN weights — CNN path removed)
- `runtime/cnn.py` (CNN inference — not imported by runtime)
- `scripts/datasets_ingest/` (ingestion scaffold)
- scikit-learn dependency (K-Means is torch-only)
- Model dropdown in UI (removed)

## Tests (105 passing)
```bash
.venv/bin/python -m pytest tests/ -q
```
Key test files:
- **Core**: test_funcs_runtime.py (21 tests)
- **Routing**: test_runtime_pipeline_routing.py, test_runtime_no_legacy_usage.py
- **K-Means**: test_kmeans_backend_routing.py, test_predict_kmeans_*.py
- **Chunks**: test_global_centers_*.py, test_label_consistency_across_chunks.py
- **Runtime modules**: test_runtime_adaptive.py, test_runtime_common.py, test_runtime_io.py, test_runtime_latent.py, test_runtime_chunking.py

## Security & Linting (QGIS recommended)
Per QGIS plugin guidelines, the plugin runtime is checked with bandit, detect-secrets, and flake8.

### Commands
```bash
# Security scan (bandit) - check for vulnerabilities
.venv/bin/bandit -r __init__.py segmenter.py segmenter_dialog.py funcs.py qgis_funcs.py \
  dependency_manager.py raster_utils.py autoencoder_utils.py runtime/ --format txt

# Secret detection - ensure no hardcoded credentials
.venv/bin/detect-secrets scan __init__.py segmenter.py segmenter_dialog.py funcs.py \
  qgis_funcs.py dependency_manager.py raster_utils.py autoencoder_utils.py runtime/*.py

# Linting (flake8)
.venv/bin/flake8 __init__.py segmenter.py segmenter_dialog.py funcs.py qgis_funcs.py \
  dependency_manager.py raster_utils.py autoencoder_utils.py runtime/ \
  --max-line-length=120 --ignore=E501,W503
```

### Expectations
- **flake8**: 0 issues (all warnings addressed or suppressed with `# noqa` for intentional patterns)
- **detect-secrets**: No secrets found (`"results": {}`)
- **bandit**: Only low-severity acceptable issues:
  - B110 (try/except/pass) — best-effort error handling for logging/callbacks
  - B404/B603 (subprocess) — required for pip bootstrap in dependency_manager.py
  - B310 (urlopen) — hardcoded https URL for get-pip.py

### Suppressed Warnings
| Code | Location | Reason |
|------|----------|--------|
| F401/F403 | segmenter.py:48 | Qt resource imports via pyrcc5 |
| E402 | segmenter.py:59-66 | Imports after `ensure_dependencies()` intentional |
| F401 | runtime/cnn.py:18 | `_DISTANCE_CHUNK_ROWS` exposed for test monkeypatching |

## Output
- uint8 label map rendered as GeoTIFF
- Preserves source extent/CRS
- Opacity 1.0
- Layer name: `{input}_{model}_{segments}_{resolution}`
