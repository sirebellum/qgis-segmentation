<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Model Notes

## Inference pipeline (current)
- Inputs: 3-band GDAL raster layer validated in [segmenter.py](segmenter.py) `_is_supported_raster_layer` (RGB GeoTIFF, provider `gdal`, band count 3). Data materialized via [funcs.py](funcs.py) `_materialize_raster` (numpy array, file path, or loader callable) which enforces `.tif/.tiff` and exact 3-band shape with clear errors.
- Runtime: backend selector in [model/runtime_backend.py](model/runtime_backend.py) chooses torch (CUDA → MPS → CPU) when available or falls back to numpy. Both backends consume the same `model/best` artifacts and expose `predict_labels`; tiled stitching remains in `predict_nextgen_numpy`.
- Output: segmentation labels as uint8 numpy array (class IDs start at 0). Rendering writes GeoTIFF via [qgis_funcs.py](qgis_funcs.py) preserving extent/CRS; opacity set to 1.0. No explicit nodata handling beyond padding trim.

## Controls, heuristics, and perf
- User input: segment count only (tile size fixed; no blur/heuristic sliders in UI).
- Dependency bootstrap: [dependency_manager.py](dependency_manager.py) installs NumPy into `vendor/` at import time unless `SEGMENTER_SKIP_AUTO_INSTALL` is set; accepts `SEGMENTER_PYTHON` for interpreter override; optional torch install is gated by `SEGMENTER_ENABLE_TORCH=1` and otherwise skipped (torch may still be used if pre-installed).
- Profiling: [perf_tuner.py](perf_tuner.py) is a shim returning default settings; no torch-prefetch or device benchmarking beyond runtime selection.

## Training (unsupervised — implemented, isolated)
- Status: eager-PyTorch pipeline in [training/](training); not wired into QGIS runtime.
- Model contract: monolithic model taking RGB only (B,3,512,512) and producing probabilities P ∈ [B,K,512,512] for K ∈ [2,16] plus stride/4 embeddings; differentiable soft k-means/EM head; fast vs learned refinement lanes.
- Losses: two-view consistency (symmetric KL on warped probabilities), entropy shaping (pixel entropy minimization + marginal entropy maximization), edge-aware smoothness using RGB gradients.
- Knobs: per-batch random K, downsample factor, cluster_iters, smooth_iters, smoothing lane; optional gradient accumulation.
- Export/Packaging: training/train.py auto-exports best checkpoint (by loss) to numpy artifacts (`model/best` + `training/best_model`) via `training/export.py` (`model.npz`, `meta.json`, `metrics.json`). The plugin ships `model/best` inside the package (see `pb_tool.cfg` `extra_dirs`) and the same directory should be uploaded as a GitHub artifact (use Git LFS if size exceeds repo limits). TorchScript export remains out-of-scope; plugin no longer consumes TorchScript models.
