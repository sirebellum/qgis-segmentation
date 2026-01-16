<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Training Baseline (Jan 2026)

- Scope (verified): current RGB-only PyTorch training scaffold, numpy export path, and offline ingestion stubs. Plugin runtime is presently the legacy TorchScript CNN/K-Means path; adoption of the new numpy runtime is **deferred** until the new model is trained.
- Sources of truth reviewed: [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md), [docs/dataset/DATASETS.md](../dataset/DATASETS.md), [docs/plugin/ARCHITECTURE.md](../plugin/ARCHITECTURE.md), [docs/plugin/MODEL_NOTES.md](../plugin/MODEL_NOTES.md), [docs/CODE_DESCRIPTION.md](../CODE_DESCRIPTION.md), [docs/AGENTIC_HISTORY.md](../AGENTIC_HISTORY.md), plus [training](../../training), [scripts/datasets_ingest](../../scripts/datasets_ingest), [scripts/data](../../scripts/data), and [model](../../model) runtime files.

## Flow (verified)
- Data prep defaults to synthetic RGB tiles from [training/data/synthetic.py](training/data/synthetic.py) wrapped by two-view augmentations in [training/data/dataset.py](training/data/dataset.py); no manifest or DEM ingestion is active.
- Training loop in [training/train.py](training/train.py) builds a loader from synthetic samples, trains `MonolithicSegmenter`, logs to TensorBoard/JSON, and auto-exports best numpy artifacts unless `--no-export` is set.
- Exports land in `model/best` (runtime pickup) and `training/best_model` (ledger mirror) via [training/export.py](training/export.py); plugin runtime consumes `model/best` through [model/runtime_numpy.py](model/runtime_numpy.py).

## Data prep & manifests
- Verified active path: synthetic-only; `DataConfig.manifest_path` exists but is unused in the loader wrapper.
- Ingestion scaffold (docs-only intent, stubbed implementation): [scripts/datasets_ingest](scripts/datasets_ingest) defines configs, provider interfaces, manifest validation, and stub providers ([providers/placeholder.py](scripts/datasets_ingest/providers/placeholder.py), [providers/naip_aws.py](scripts/datasets_ingest/providers/naip_aws.py)). CLI in [scripts/datasets_ingest/cli.py](scripts/datasets_ingest/cli.py) lists providers and emits placeholder manifest plans; no IO/network/GDAL.
- Legacy configs: [configs/datasets/naip_3dep_example.yaml](configs/datasets/naip_3dep_example.yaml) and [configs/datasets/naip_aws_3dep_example.yaml](configs/datasets/naip_aws_3dep_example.yaml) reference elevation fields (`standardize_elevation`, etc.) not present in [training/config.py](training/config.py); these are not consumed.
- Legacy NAIP/DEM prep scripts referenced in history (e.g., `prepare_naip_aws_3dep_dataset.py`) are absent; only GDAL helpers remain in [scripts/data/_gdal_utils.py](scripts/data/_gdal_utils.py).

## Config system
- Default config dataclasses live in [training/config.py](training/config.py); knobs include `max_k<=16`, stride/4 encoder, clustering/smoothing ranges, batch/steps, and augmentation choices.
- [training/config_loader.py](training/config_loader.py) loads python config modules exposing `get_config()` or `CONFIG`; YAML is optional (PyYAML required). Overrides for steps/grad_accum/max_samples are applied from CLI flags in [training/train.py](training/train.py).

## Training entrypoints
- Train: `python -m training.train --steps 2 --amp 0 --checkpoint_dir /tmp/seg_ckpt --no-export` (CPU-friendly smoke). Uses Adam, optional CUDA AMP, gradient accumulation, and random `k`/downsample/cluster/smooth knobs via `sample_knobs()`.
- Eval: `python -m training.eval --synthetic --seed 7` runs proxy metrics (utilization, speckle, boundary density, self-consistency) on synthetic samples with `k<=4`.
- Dataloading: `UnsupervisedRasterDataset` augments each RGB sample into two stochastic views (flip/rotate/noise) and emits an identity warp grid; values normalized to [0,1].

## Export & runtime contract
- Exported artifacts from [training/export.py](training/export.py):
  - `model.npz` with weights renamed for the runtime stem/block1/block2/seed_proj layout.
  - `meta.json` fields: `version`, `max_k`, `embed_dim`, `temperature`, `cluster_iters_default`, `smooth_iters_default`, `input_mean/std/scale`, `stride=4`, `supports_learned_refine=false`.
  - `metrics.json` with `score` (best loss proxy) and `step` plus optional extras.
- Runtime consumer: [model/runtime_numpy.py](model/runtime_numpy.py) loads `model/best`, pads to stride, runs encoder + soft k-means + optional depthwise smoothing, upsamples, and returns labels via `predict_labels`. Learned refinement is not implemented in the runtime; only fast smoothing is applied. **The QGIS plugin does not yet consume these artifacts; runtime integration is deferred.**

## Tests (offline by default)
- Numpy runtime/export: [tests/test_export_to_numpy_runtime.py](tests/test_export_to_numpy_runtime.py), [tests/test_numpy_runtime_tiling.py](tests/test_numpy_runtime_tiling.py).
- Training shapes/losses/smoke: [training/tests/test_model_shapes.py](training/tests/test_model_shapes.py), [training/tests/test_losses.py](training/tests/test_losses.py), [training/tests/test_smoke_train.py](training/tests/test_smoke_train.py).
- GDAL helpers: [tests/test_alignment_invariants.py](tests/test_alignment_invariants.py).
- Ingestion scaffold stubs: [tests/test_datasets_ingest_stub.py](tests/test_datasets_ingest_stub.py).
- Optional QGIS smoke skipped unless `QGIS_TESTS=1`: [tests/test_qgis_runtime_smoke.py](tests/test_qgis_runtime_smoke.py).

## Known gaps / doc ↔ code mismatches
- Manifest-backed real data ingestion is disabled; scaffold exists but training uses only synthetic tensors.
- Legacy NAIP/DEM prep scripts cited in history are missing; ingestion rewrite is not yet implemented beyond stubs.
- Sample dataset configs carry elevation keys that the current `DataConfig` ignores.
- Training supports a learned refinement head, but runtime exports flag `supports_learned_refine=false` and the numpy runtime applies only fast smoothing.
