<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# CODE_DESCRIPTION

Purpose: concise registry of modules and their responsibilities in the current codebase.

## Plugin Runtime (execute_* pipeline)
- [segmenter.py](../segmenter.py), [segmenter_dialog.py](../segmenter_dialog.py), [segmenter_dialog_base.ui](../segmenter_dialog_base.ui): QGIS UI + task dispatch; validates 3-band GDAL GeoTIFFs and segment count; builds heuristic/post-smoothing overrides and queues `execute_cnn_segmentation` or `execute_kmeans_segmentation` via `QgsTask`.
- [funcs.py](../funcs.py): backward-compatibility facade that re-exports the numerical engine from the `runtime/` package.
- [runtime/](../runtime): split numerical engine with modules for:
  - `common.py`: cancellation tokens, status callbacks, device coercion, dtype helpers
  - `io.py`: raster/model materialization from paths or callables
  - `adaptive.py`: chunk planning, memory budgets, tile sizing
  - `chunking.py`: chunk aggregation, label normalization, one-hot conversion
  - `distance.py`: pairwise distance kernels, chunked argmin, bruteforce KNN
  - `smoothing.py`: Gaussian blur channels, weight masks
  - `latent.py`: latent KNN soft refinement, stratified sampling
  - `kmeans.py`: torch-only K-Means with global center fit + streaming assignment
  - `cnn.py`: CNN tiling, prefetch, global center fit, inference with centers
  - `pipeline.py`: high-level `execute_cnn_segmentation`, `execute_kmeans_segmentation`, optional post-blur, texture autoencoder hooks
- [models/](../models): TorchScript CNN weights (`model_4.pth`, `model_8.pth`, `model_16.pth`) loaded by `segmenter.load_model`.
- [qgis_funcs.py](../qgis_funcs.py): GDAL GeoTIFF rendering + layer registration.
- [dependency_manager.py](../dependency_manager.py): on-demand install of torch/NumPy into `vendor/` with env overrides; no scikit-learn required.
- [raster_utils.py](../raster_utils.py): `ensure_channel_first` helper for array reordering.
- [autoencoder_utils.py](../autoencoder_utils.py): texture autoencoder manager for optional remap (disabled by default in plugin).

## Training
- [training/train.py](../training/train.py): legacy entrypoint that forwards to `train_distill.py`.
- [training/train_distill.py](../training/train_distill.py): unified distillation trainer supporting:
  - Optional teacher (DINOv2 or fake fallback, off by default)
  - Shard-backed GeoTIFF patches with precomputed SLIC labels
  - Losses: feature/affinity distillation (optional), clustering shaping, SLIC boundary priors, edge-aware TV, EMA-normalized scale-neutral merge
  - Autoencoder reconstruction loss (enabled by default): low-pass RGB + gradient consistency
- [training/eval.py](../training/eval.py): evaluation runner for proxy metrics.
- [training/export.py](../training/export.py): exports MonolithicSegmenter checkpoints to numpy artifacts (used by tests, not yet consumed by plugin).
- [training/config.py](../training/config.py), [training/config_loader.py](../training/config_loader.py): dataclass configs with defaults; supports python module overrides.
- [training/data/](../training/data): data loaders:
  - `synthetic.py`: RGB-only synthetic samples for offline tests
  - `sharded_tif_dataset.py`: iterable shard loader with worker partitioning and LRU caching
  - `geo_patch_dataset.py`: GeoTIFF random-window sampler for RGB patches
  - `dataset.py`: two-view augmentation wrapper
- [training/models/](../training/models): model architectures:
  - `model.py`: MonolithicSegmenter (encoder + soft k-means head + refiner)
  - `backbone.py`: stride-4 encoder
  - `soft_cluster.py`: differentiable soft k-means head
  - `refine.py`: learned refiner + fast smoothing
  - `student_cnn.py`: single-scale stride-4 student for distillation
- [training/teachers/](../training/teachers): teacher adapters (training-only):
  - `teacher_base.py`: base class + FakeTeacher
  - `dinov2.py`: DINOv2 wrapper
- [training/losses.py](../training/losses.py): unsupervised losses (consistency, entropy, smoothness).
- [training/losses_distill.py](../training/losses_distill.py): distillation losses (feature/affinity, clustering, boundary priors, TV).
- [training/losses_recon.py](../training/losses_recon.py): training-only reconstruction loss with TinyReconDecoder.
- [training/metrics.py](../training/metrics.py): proxy metrics (utilization, speckle, boundary, consistency).
- [training/augmentations.py](../training/augmentations.py): synchronized geometry + photometric transforms for RGB/SLIC/targets.
- [training/utils/](../training/utils): helpers (seed, warp, gradients, resample).
- [training/tests/](../training/tests): pytest coverage (24+ test files covering model shapes, losses, loaders, distillation, etc.).

## Dataset Prep
- [training/datasets/header_schema.py](../training/datasets/header_schema.py): YAML schema + validator for dataset headers.
- [training/datasets/generate_headers.py](../training/datasets/generate_headers.py): scanner to emit headers from extracted data.
- [training/datasets/build_shards.py](../training/datasets/build_shards.py): shard builder producing uncompressed GeoTIFFs with `index.jsonl`, deterministic splits, superpixel precompute (SLIC/SEEDS with grid fallback).
- [training/datasets/metrics.py](../training/datasets/metrics.py): IoU helper with label masking.
- [training/datasets/headers/](../training/datasets/headers): generated YAML headers per dataset.
- [training/datasets/tests/](../training/datasets/tests): header/shard tests.

## Tests (repo-level)
- [tests/](../tests): pytest suite (21+ test files) covering:
  - Runtime pipeline routing, global centers, label consistency
  - K-Means backend routing, determinism, memory bounds, GPU smoke
  - Distance utils, alignment invariants
  - Plugin imports (skip-gated for missing packages)
  - QGIS smoke (skip-gated unless `QGIS_TESTS=1`)

## Key Contracts
- **Input validation**: 3-band GDAL GeoTIFF (`.tif/.tiff`), provider `gdal`, enforced in `segmenter._is_supported_raster_layer`.
- **K-Means**: torch-only with global center fit + streaming assignment (no per-chunk relabeling, no scikit-learn).
- **CNN**: TorchScript models with global center fit + chunked assignment via `_process_in_chunks`.
- **Training entrypoint**: `python -m training.train` forwards to `python -m training.train_distill`.
- **Autoencoder**: enabled by default in training (`autoencoder.enabled=true`), decoder excluded from deployment.
- **SLIC**: precomputed during shard build; loader requires SLIC by default (`data.require_slic=true`).

## Env Overrides
- `SEGMENTER_SKIP_AUTO_INSTALL`: skip vendor dependency install
- `SEGMENTER_PYTHON`, `SEGMENTER_PIP_EXECUTABLE`: interpreter/pip overrides
- `SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`: torch wheel selection

## Notes
- The `model/` directory (numpy runtime selectors) and `scripts/datasets_ingest/` (ingestion scaffold) referenced in older docs have been removed from the repository.
- Tests that depend on these missing packages are skipped.
- Plugin runtime uses TorchScript CNN + torch K-Means; next-gen artifacts from training export are not yet consumed.
