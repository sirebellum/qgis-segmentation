<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Model Notes

## Inference pipeline (current)
- Inputs: 3-band GDAL raster layer validated in [segmenter.py](segmenter.py) `_is_supported_raster_layer` (RGB GeoTIFF, provider `gdal`, band count 3). Data materialized via [funcs.py](funcs.py) `_materialize_raster` (numpy array, file path, or loader callable).
- K-Means path: `predict_kmeans` pads to block `resolution`, reshapes to block vectors, samples up to ~10k blocks (scaled by accuracy slider), fits `sklearn.KMeans`, assigns clusters (GPU-assisted if available), upsamples to full resolution, then optional blur (`blur_segmentation_map`).
- CNN path: TorchScript model loaded in `Segmenter.load_model()` from `models/model_<resolution>.pth`; `predict_cnn` tiles raster (`tile_raster`), batches tiles with adaptive batch size/prefetch (`_recommended_batch_size`, `_prefetch_batches`), runs model forward (expects tuple where index 1 is latent feature grid). Latent grid clustered via `predict_kmeans` at resolution 1; optional `latent_knn_soft_refine` smooths labels using latent vectors + spatial coords. Tile grid reconstructed, auto-oriented (`_auto_orient_tile_grid`), optionally rotates score volume if scores requested. Optional blur applied post-hoc.
- Output: segmentation labels as uint8 numpy array (class IDs start at 0). Rendering writes GeoTIFF via [qgis_funcs.py](qgis_funcs.py) preserving extent/CRS; opacity set to 1.0. No explicit nodata handling beyond padding trim.

## Controls, heuristics, and perf
- Resolution presets: `RESOLUTION_CHOICES` label → value map; speed slider scales effective resolution for K-Means sampling density and CNN heuristic overrides.
- Blur: `_legacy_blur_config` derives kernel/iterations from smoothness slider; `_apply_legacy_blur` runs depthwise convolutional smoothing in torch.
- Latent KNN: defaults in `LATENT_KNN_DEFAULTS`; overrides built in `_build_heuristic_overrides` adjust neighbors, temperature, mix, spatial weight, chunk sizes, hierarchy passes based on sliders/resolution.
- GPU/VRAM heuristics: `_quant_chunk_size` uses fractions of free CUDA/MPS/system memory to size GPU cluster batches; `_recommended_batch_size` respects `memory_budget` and prefetch depth (`AdaptiveSettings`).
- Profiling: `load_or_profile_settings` (perf_profile.json in plugin root) benchmarks throughput once per device to set safety factor/prefetch depth per tier; gated by `SEGMENTER_SKIP_PROFILING` env. Status surfaced through `Segmenter.log_status`.
- Dependency bootstrap: [dependency_manager.py](dependency_manager.py) installs torch/numpy/scikit-learn into `vendor/` at import time unless `SEGMENTER_SKIP_AUTO_INSTALL` is set; accepts `SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`, `SEGMENTER_PYTHON` for overrides.

## Training (Planned — unsupervised)
- Status: not implemented; current models are TorchScript binaries shipped in `models/`.
- Data: unlabeled imagery only; assume 3-channel RGB input; any labels for evaluation remain optional.
- Model export contract: TorchScript returning `(mask, features)` or `(None, features)` compatible with `predict_cnn` latent usage (see training/README.md).
- TODO:
  - Populate training/data layout and config specs in `training/` (unlabeled inputs, optional eval labels).
  - Define augmentation strategy suitable for unlabeled data (e.g., color jitter, random crops/rotations).
  - Add evaluation hooks for proxy metrics (e.g., feature diversity, clustering stability) once design settles.
  - Record architecture/loss choices in `training/MODEL_HISTORY.md` as experiments proceed.
