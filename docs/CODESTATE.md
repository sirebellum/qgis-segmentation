<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# CODESTATE

## Overview
QGIS plugin for map segmentation + isolated PyTorch training/export stack and dataset prep. Plugin runtime uses TorchScript CNN + torch-only K-Means with global center fit and chunk aggregation. Training produces student embeddings with optional distillation; autoencoder reconstruction loss enabled by default.

## Path Map
### Plugin Runtime
- [segmenter.py](../segmenter.py), [segmenter_dialog.py](../segmenter_dialog.py), [segmenter_dialog_base.ui](../segmenter_dialog_base.ui): UI + task dispatch
- [qgis_funcs.py](../qgis_funcs.py): GDAL render + layer registration
- [dependency_manager.py](../dependency_manager.py): torch/numpy vendor install
- [funcs.py](../funcs.py): facade re-exporting runtime/ symbols
- [runtime/](../runtime): numerical engine (pipeline.py, kmeans.py, cnn.py, chunking.py, etc.)
- [models/](../models): TorchScript weights (model_4.pth, model_8.pth, model_16.pth)
- [autoencoder_utils.py](../autoencoder_utils.py): texture remap manager (optional)
- [raster_utils.py](../raster_utils.py): array channel ordering

### Training
- [training/train.py](../training/train.py): legacy entrypoint → forwards to train_distill.py
- [training/train_distill.py](../training/train_distill.py): unified distillation trainer
- [training/eval.py](../training/eval.py): evaluation runner
- [training/export.py](../training/export.py): numpy artifact export
- [training/config.py](../training/config.py): dataclass configs
- [training/data/](../training/data): synthetic.py, sharded_tif_dataset.py, geo_patch_dataset.py, dataset.py
- [training/models/](../training/models): model.py, backbone.py, soft_cluster.py, refine.py, student_cnn.py
- [training/teachers/](../training/teachers): teacher_base.py, dinov2.py (training-only)
- [training/losses.py](../training/losses.py), [losses_distill.py](../training/losses_distill.py), [losses_recon.py](../training/losses_recon.py)
- [training/metrics.py](../training/metrics.py), [augmentations.py](../training/augmentations.py)
- [training/utils/](../training/utils): seed.py, warp.py, resample.py, gradients.py

### Dataset Prep
- [training/datasets/header_schema.py](../training/datasets/header_schema.py): YAML schema + validation
- [training/datasets/generate_headers.py](../training/datasets/generate_headers.py): header generation
- [training/datasets/build_shards.py](../training/datasets/build_shards.py): shard builder with SLIC precompute
- [training/datasets/metrics.py](../training/datasets/metrics.py): IoU with label masking
- [training/datasets/headers/](../training/datasets/headers): generated YAML headers

### Tests
- [tests/](../tests): 21+ files (runtime, K-Means, CNN, distance, plugin imports)
- [training/tests/](../training/tests): 24+ files (model, losses, loaders, distillation, autoencoder)
- [training/datasets/tests/](../training/datasets/tests): header/shard validation

## Entrypoints
```bash
# Training (unified entrypoint)
python -m training.train --synthetic --steps 3 --seed 123

# Distillation (direct)
python -m training.train_distill --synthetic --steps 3 --seed 123

# Evaluation
python -m training.eval --synthetic --seed 7

# Shard generation
python -m training.datasets.generate_headers --dry-run
python -m training.datasets.build_shards --overwrite --seed 123 --shard-size 512

# Tests
.venv/bin/python -m pytest -q
```

## Runtime Contracts
- **Inputs**: 3-band GDAL GeoTIFF (.tif/.tiff), provider=gdal, validated in `segmenter._is_supported_raster_layer`
- **K-Means**: torch-only, global center fit, streaming assignment (no scikit-learn)
- **CNN**: TorchScript models loaded via `segmenter.load_model`, global centers + chunked assignment
- **Output**: uint8 label map rendered as GeoTIFF via `qgis_funcs.render_raster`

## Training Contracts
- **Student**: single stride-4 embedding path, input-size invariant across patch sizes (256/512/1024)
- **Autoencoder**: enabled by default (`autoencoder.enabled=true`), decoder excluded from deployment
- **SLIC**: precomputed during shard build, required by loader (`data.require_slic=true`)
- **Losses**: feature/affinity distillation (optional), clustering shaping, boundary priors, edge-aware TV, reconstruction
- **Targets**: used only for IoU metrics, labels ≤0 masked

## Key Config Knobs
### Training (config.py)
- `model.embed_dim`, `model.max_k`, `model.temperature`
- `data.patch_sizes`, `data.source` (synthetic/shards), `data.require_slic`
- `autoencoder.enabled`, `autoencoder.lambda_recon`, `autoencoder.blur_sigma`
- `student.embed_dim`, `student.depth`, `student.norm`
- `distill.cluster_iters`, `distill.ema_decay`
- `train.multi_scale_mode` (sample_one_scale_per_step / per_step_all_scales)

### Plugin (env vars)
- `SEGMENTER_SKIP_AUTO_INSTALL`: skip vendor install
- `SEGMENTER_PYTHON`, `SEGMENTER_PIP_EXECUTABLE`: interpreter/pip
- `SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`: torch wheel selection

## Removed / Non-Existent
The following paths referenced in older docs do not exist:
- `model/` directory (runtime_numpy.py, runtime_backend.py, runtime_torch.py)
- `scripts/datasets_ingest/` (ingestion scaffold)
- `configs/datasets/` (legacy NAIP/3DEP configs)

Tests that import from these paths are skipped.

## Test Inventory (current)
### tests/ (plugin runtime) — 91 tests passing
- **Core runtime**: test_funcs_runtime.py (21 tests)
- **Global centers**: test_global_centers_assignment.py, test_global_centers_canonical.py
- **K-Means routing**: test_kmeans_backend_routing.py, test_predict_kmeans_*.py
- **Chunk processing**: test_label_consistency_across_chunks.py, test_no_legacy_chunk_fit.py, test_no_per_chunk_kmeans_fit.py
- **Pipeline routing**: test_runtime_pipeline_routing.py, test_runtime_no_legacy_usage.py
- **Distance utilities**: test_torch_distance_utils.py
- **Runtime modules**:
  - test_runtime_adaptive.py (15 tests) — AdaptiveSettings, ChunkPlan, memory probes
  - test_runtime_common.py (12 tests) — device coercion, cancellation, status emit
  - test_runtime_io.py (6 tests) — raster/model materialization
  - test_runtime_latent.py (8 tests) — resize, stratified sampling, KNN refinement
  - test_runtime_chunking.py (10 tests) — chunk starts, one-hot, aggregator
- **Skip-gated**:
  - test_plugin_imports.py (requires QGIS context)
  - test_qgis_runtime_smoke.py (set QGIS_TESTS=1)
  - test_predict_kmeans_gpu_smoke.py (set RUN_GPU_TESTS=1)

### training/tests/ — 82 tests passing
- **Models**: test_model_shapes.py, test_model_forward_contract.py
- **Student**: test_student_embed.py
- **Losses**: test_losses.py, test_distill_losses_smoke.py, test_multires_losses.py
- **Autoencoder**: test_autoencoder_losses.py (27 tests)
- **Boundary priors**: test_boundary_priors.py, test_loss_slic_priors.py
- **Datasets**: test_sharded_dataset_*.py, test_geo_patch_dataset.py
- **Augmentations**: test_augmentations.py
- **Smoke train**: test_smoke_train.py
- **Misc**: test_synthetic.py, test_teacher_fallback.py, test_knobs_sampling.py, test_metrics_*.py

## Coverage Summary
- **173 total tests** (91 plugin + 82 training)
- Plugin tests avoid QGIS/GDAL imports (offline-safe)
- Training tests use synthetic data by default
- GPU tests gated behind `RUN_GPU_TESTS=1`
