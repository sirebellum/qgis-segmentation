<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Model Notes

## Inference pipeline (current runtime)
- Inputs: 3-band GDAL raster layer validated in [segmenter.py](segmenter.py) `_is_supported_raster_layer` (RGB GeoTIFF, provider `gdal`, band count 3). Data materialized via [funcs.py](funcs.py) `_materialize_raster` (numpy array, file path, or loader callable) which loads through GDAL when given a path.
- Runtime: user selects **CNN** (TorchScript) or **K-Means** in the UI. The plugin now routes to `execute_cnn_segmentation` / `execute_kmeans_segmentation`, which compute adaptive chunk plans, tile/aggregate inference, and apply optional post-smoothing. **CNN and K-Means both fit global centers once per run and stream assignment per chunk (no per-chunk relabeling)** to keep IDs consistent. Texture remap (autoencoder) is available but disabled in the plugin by default.
- Output: segmentation labels as uint8 numpy array (class IDs start at 0). Rendering writes GeoTIFF via [qgis_funcs.py](qgis_funcs.py) preserving extent/CRS; opacity set to 1.0. No explicit nodata handling beyond padding trim.

## Controls, heuristics, and perf
- User input: segment count, resolution choice, and sliders for smoothness/speed/accuracy. Sliders influence blur kernel/iterations, CNN tile size (192–768px), sampling scale, and latent KNN overrides.
- Dependency bootstrap: [dependency_manager.py](dependency_manager.py) installs torch and NumPy into `vendor/` at import time unless `SEGMENTER_SKIP_AUTO_INSTALL` is set; interpreter override via `SEGMENTER_PYTHON`; torch index/spec knobs (`SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`).
- Adaptive settings: static defaults (prefetch depth 2, safety factor 8) from `AdaptiveSettings`; no device profiling beyond the CUDA → MPS → CPU device pick in `segmenter.py`.

## Training (unsupervised — implemented, isolated)
- Status: eager-PyTorch pipeline in [training/](training); exports numpy artifacts for a **future** runtime path, not consumed by the current plugin.
- Model contract: monolithic model taking RGB only (B,3,512,512) and producing probabilities P ∈ [B,K,512,512] for K ∈ [2,16] plus stride/4 embeddings; differentiable soft k-means/EM head; fast vs learned refinement lanes.
- Losses: two-view consistency (symmetric KL on warped probabilities), entropy shaping (pixel entropy minimization + marginal entropy maximization), edge-aware smoothness using RGB gradients.
- Knobs: per-batch random K, downsample factor, cluster_iters, smooth_iters, smoothing lane; optional gradient accumulation.
- Export/Packaging: `training/train.py` auto-exports best checkpoint (by loss) to numpy artifacts (`model/best` + `training/best_model`) via `training/export.py` (`model.npz`, `meta.json`, `metrics.json`). The plugin currently ships TorchScript models under `models/`; updating the runtime to consume the new artifacts is **deferred until training completes**.
