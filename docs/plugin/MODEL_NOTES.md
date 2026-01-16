<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Model Notes

## Inference pipeline (current)
- Inputs: 3-band GDAL raster layer validated in [segmenter.py](segmenter.py) `_is_supported_raster_layer` (RGB GeoTIFF, provider `gdal`, band count 3). Data materialized via [funcs.py](funcs.py) `_materialize_raster` (numpy array, file path, or loader callable).
- Runtime: numpy-only path via `predict_nextgen_numpy` (tiles/stitches with padding trim) and `model/runtime_numpy.py` (loads `model/best` artifacts and exposes `predict_labels`). No torch/scikit-learn dependency in the plugin runtime.
- Output: segmentation labels as uint8 numpy array (class IDs start at 0). Rendering writes GeoTIFF via [qgis_funcs.py](qgis_funcs.py) preserving extent/CRS; opacity set to 1.0. No explicit nodata handling beyond padding trim.

## Controls, heuristics, and perf
- User input: segment count only (tile size fixed; no blur/heuristic sliders in UI).
- Dependency bootstrap: [dependency_manager.py](dependency_manager.py) installs NumPy into `vendor/` at import time unless `SEGMENTER_SKIP_AUTO_INSTALL` is set; accepts `SEGMENTER_PYTHON` for interpreter override.
- Profiling: [perf_tuner.py](perf_tuner.py) is a shim returning default settings; no torch imports or device benchmarking.

## Training (unsupervised — implemented, isolated)
- Status: eager-PyTorch pipeline in [training/](training); not wired into QGIS runtime.
- Model contract: monolithic model taking RGB (B,3,512,512) and optional elevation (B,1,512,512) with per-sample masks, producing probabilities P ∈ [B,K,512,512] for K ∈ [2,16] and embeddings stride/4 (B,D,128,128); FiLM-like elevation injection; differentiable soft k-means/EM head; fast vs learned refinement lanes.
- Losses: two-view consistency (symmetric KL on warped probabilities), entropy shaping (pixel entropy minimization + marginal entropy maximization), edge-aware smoothness weighted by RGB and optional elevation gradients.
- Knobs: per-batch random K, downsample factor, cluster_iters, smooth_iters, smoothing lane; elevation dropout even when elevation is present; optional gradient accumulation.
- Export: training/train.py auto-exports best checkpoint (by loss) to numpy artifacts (`model/best` + `training/best_model`) via `training/export.py` (`model.npz`, `meta.json`, `metrics.json`). TorchScript export remains out-of-scope; plugin no longer consumes TorchScript models.
