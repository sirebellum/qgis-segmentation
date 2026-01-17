<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Agentic History

New series beginning at the current repository state. Phase numbering restarts here; Phase 0 captures the present code snapshot and invariants.

## Phase 0 — Current State (Reset)
- **Runtime**: QGIS plugin runs a single numpy-only path. `Segmenter.predict` in [segmenter.py](segmenter.py#L620-L697) queues `predict_nextgen_numpy` from [funcs.py](funcs.py#L32-L88) with tiling/stitching and cancellation. Rendering writes GeoTIFF via [qgis_funcs.py](qgis_funcs.py#L7-L45). Dependency bootstrap is NumPy-only in [dependency_manager.py](dependency_manager.py#L1-L204); adaptive settings used static defaults (no profiling shim). UI stripped to layer picker + segment count ([segmenter_dialog_base.ui](segmenter_dialog_base.ui)).
- **Model runtime**: Loads numpy artifacts from `model/best` through [model/runtime_numpy.py](model/runtime_numpy.py) (consumes `model.npz` + `meta.json`), exposes `predict_labels` for tiled inference; no torch imports on the runtime path.
- **Training**: Eager PyTorch scaffolding remains RGB-only. Synthetic dataset + paired-view augmentations in [training/data/dataset.py](training/data/dataset.py) and [training/data/synthetic.py](training/data/synthetic.py); model/losses in [training/models](training/models) and [training/losses.py](training/losses.py); CLI runners in [training/train.py](training/train.py) and [training/eval.py](training/eval.py); exports numpy artifacts via [training/export.py](training/export.py). Real-world dataset ingestion is paused; NAIP/DEM loaders have been removed.
- **Datasets**: Historical configs remain in [configs/datasets](configs/datasets), but active ingestion code is limited to GDAL helpers ([scripts/data/_gdal_utils.py](scripts/data/_gdal_utils.py)) and fixtures. Dataset ingestion is flagged for rewrite; default training relies on synthetic data only.
- **Tests**: Offline pytest suite covers numpy runtime tiling/export and alignment invariants ([tests](tests)). QGIS smoke test is optional/skipped by default (`QGIS_TESTS=1`).
- **Docs/Invariants**: Architecture and model notes live under [docs/plugin](plugin). Training pipeline (RGB-only) in [docs/training/TRAINING_PIPELINE.md](training/TRAINING_PIPELINE.md). Required inputs listed in [docs/AGENTIC_REQUIRED.md](AGENTIC_REQUIRED.md). SPDX: BSD-3-Clause across code/docs. Plugin runtime must remain torch-free and QGIS-free for default tests.

### Phase template
- **Intent**:
- **Summary**:
- **Files Touched**:
- **Commands**:
- **Validation**:
- **Risks/Notes**:

## Phase 1 — Generated runtime status overview
- **Intent**: Capture the current QGIS runtime state in a concise doc for prompt scaffolding.
- **Summary**: Added [docs/plugin/RUNTIME_STATUS.md](plugin/RUNTIME_STATUS.md) describing UI→task→numpy path, model contract, config points, logging/cancel flows, and gaps; linked [docs/plugin/ARCHITECTURE.md](plugin/ARCHITECTURE.md) to it; recorded phase in [docs/CODE_DESCRIPTION.md](CODE_DESCRIPTION.md).
- **Files Touched**: [docs/plugin/RUNTIME_STATUS.md](plugin/RUNTIME_STATUS.md), [docs/plugin/ARCHITECTURE.md](plugin/ARCHITECTURE.md), [docs/CODE_DESCRIPTION.md](CODE_DESCRIPTION.md), [docs/AGENTIC_HISTORY.md](AGENTIC_HISTORY.md).
- **Commands**: none (doc-only).
- **Validation**: Manual link sanity checks (internal doc links); no tests executed.
- **Risks/Notes**: Snapshot reflects current workspace only; unused torch-prefetch helpers remain in funcs.py but are not on the plugin path.

## Phase 2 — Training baseline snapshot
- **Intent**: Capture a repo-verified baseline of the current training pipeline, export contract, and ingestion scaffold status.
- **Summary**: Added [docs/training/TRAINING_BASELINE.md](training/TRAINING_BASELINE.md) covering synthetic-only data flow, config/CLI entrypoints, numpy export/runtime contract, tests, and documented code-vs-doc gaps; fixed dataset doc reference in [docs/CODE_DESCRIPTION.md](CODE_DESCRIPTION.md).
- **Files Touched**: [docs/training/TRAINING_BASELINE.md](training/TRAINING_BASELINE.md), [docs/CODE_DESCRIPTION.md](CODE_DESCRIPTION.md), [docs/AGENTIC_HISTORY.md](AGENTIC_HISTORY.md).
- **Commands**: none (doc-only).
- **Validation**: Link sanity checks for new baseline doc; no tests executed.
- **Risks/Notes**: Dataset ingestion remains stubbed; legacy NAIP/DEM prep scripts referenced historically are still absent.

## Phase 3 — Runtime invariants hardening

## Phase 4 — Runtime contract sync + smoke export


## Phase 6 — Import/package hardening

## Phase 7 — Torch bootstrap knobs

## Phase 8 — Sync runtime docs to restored original runtime
- **Intent**: Align runtime-facing docs with the restored legacy TorchScript CNN + scikit-learn K-Means code and clarify that the new model runtime integration is deferred.
- **Summary**: Reconciled ARCHITECTURE, MODEL_NOTES, RUNTIME_STATUS, CODE_DESCRIPTION, and AGENTIC_HISTORY against the actual plugin code; marked historical runtime selector/import/torch-bootstrap phases as superseded; reiterated that next-gen numpy runtime is deferred until training finishes.
- **Files Touched**: [docs/plugin/ARCHITECTURE.md](plugin/ARCHITECTURE.md), [docs/plugin/MODEL_NOTES.md](plugin/MODEL_NOTES.md), [docs/plugin/RUNTIME_STATUS.md](plugin/RUNTIME_STATUS.md), [docs/CODE_DESCRIPTION.md](docs/CODE_DESCRIPTION.md), [docs/AGENTIC_HISTORY.md](AGENTIC_HISTORY.md).
- **Commands**: `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m compileall .` (pass); `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m pytest -q` (fails during collection: `ModuleNotFoundError: No module named 'qgis'` from runtime tests importing segmenter.py). No remediation in this doc-only pass.
- **Validation**: Compileall succeeded. Pytest requires QGIS bindings for runtime tests; documented failure and left unresolved per scope.
- **Risks/Notes**: Runtime update for the new model type remains deferred until training is complete. Default test suite currently depends on QGIS for runtime modules; consider gating or stubbing if QGIS-free runs are required in future iterations.

## Phase 23 — Dataset prep v0 (headers + shards)
- **Intent**: Stand up the dataset header schema, ms_buildings header generation, deterministic shard builder, and masked IoU helper while keeping tests offline/QGIS-free.
- **Summary**: Added header validator/schema, ms_buildings header, generator CLI (with extracted-root fallback), shard builder producing uncompressed GeoTIFF shards plus index.jsonl and summary manifest, masked IoU helper, and offline tests covering schema parsing, split logic, shard layout, and IoU masking. Documented schema and pipeline references.
- **Files Touched**: [training/datasets/header_schema.py](training/datasets/header_schema.py), [training/datasets/generate_headers.py](training/datasets/generate_headers.py), [training/datasets/build_shards.py](training/datasets/build_shards.py), [training/datasets/metrics.py](training/datasets/metrics.py), [training/datasets/headers/ms_buildings.yaml](training/datasets/headers/ms_buildings.yaml), [training/datasets/tests/test_pipeline.py](training/datasets/tests/test_pipeline.py), [docs/dataset/HEADERS.md](docs/dataset/HEADERS.md), [docs/dataset/DATASETS.md](docs/dataset/DATASETS.md), [docs/training/TRAINING_PIPELINE.md](docs/training/TRAINING_PIPELINE.md), [docs/CODE_DESCRIPTION.md](docs/CODE_DESCRIPTION.md), [requirements.txt](requirements.txt).
- **Commands**: `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m training.datasets.generate_headers --dataset ms_buildings --dry-run`; `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m training.datasets.build_shards --dataset-id ms_buildings --dry-run --max-items 5`; `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m pytest -q`.
- **Validation**: Pytest passed (57 passed, 5 skipped; warnings about custom gpu mark and rasterio non-georeferenced test fixtures). Dry-run header/shard commands verified ms_buildings counts and fallback roots.
- **Risks/Notes**: Tools default to `training/datasets/extracted` but fall back to `training/datasets/data/extracted`; keep this in sync with future data layout. Shard builder enforces overwrite guard; outputs are uncompressed GeoTIFF. Metrics split sends 25% of labeled tiles to `metrics_train` for metrics-only evaluation.

## Phase 24 — Sharded ingestion wired into training
- **Intent**: Activate processed GeoTIFF shard ingestion in training/eval with parallel loading, optional caching, and metrics-only labels while keeping torch-free runtime untouched.
- **Summary**: Added IterableDataset for shard dirs with worker partitioning and optional per-worker LRU cache, normalized label remap and RGB tensors, and paired-view aug support. Wired shard source into train/eval configs/CLI (dataset id, processed root, eval interval, perf knobs) and routed metrics through masked IoU ignoring labels <=0. Updated docs to reflect shard ingestion availability and recorded model history entry. Fixed tests to cover loader partitioning/caching and IoU masking rules.
- **Files Touched**: [training/data/sharded_tif_dataset.py](training/data/sharded_tif_dataset.py), [training/data/__init__.py](training/data/__init__.py), [training/config.py](training/config.py), [training/train.py](training/train.py), [training/eval.py](training/eval.py), [training/tests/test_sharded_dataset_loader.py](training/tests/test_sharded_dataset_loader.py), [training/tests/test_metrics_iou_ignore_zero.py](training/tests/test_metrics_iou_ignore_zero.py), [docs/dataset/DATASETS.md](docs/dataset/DATASETS.md), [docs/training/TRAINING_PIPELINE.md](docs/training/TRAINING_PIPELINE.md), [docs/training/TRAINING_BASELINE.md](docs/training/TRAINING_BASELINE.md), [docs/training/MODEL_HISTORY.md](docs/training/MODEL_HISTORY.md), [docs/CODE_DESCRIPTION.md](docs/CODE_DESCRIPTION.md), [docs/AGENTIC_HISTORY.md](docs/AGENTIC_HISTORY.md).
- **Commands**: `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m compileall training`; `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m pytest -q` (66 passed, 5 skipped; rasterio/georef warnings only).
- **Validation**: Compileall + full pytest suite passed; multiworker DataLoader test confirmed partitioning once collate function was made picklable.
- **Risks/Notes**: DataLoader perf knobs default to conservative values; large shard sets may need tuned `num_workers`/`prefetch_factor`. Metrics ignore labels <=0; ensure shard targets use positive IDs for evaluable classes. Runtime path unchanged (still torchscript CNN/KMeans legacy).

## Phase 25 — Doc hygiene (remove missing NAIP/3DEP references)
- **Intent**: Align docs with repository contents by removing references to deleted NAIP/3DEP ingestion scripts/providers.
- **Summary**: Marked historical NAIP/3DEP sections in [docs/CODE_DESCRIPTION.md](docs/CODE_DESCRIPTION.md) as removed and clarified that only offline stubs remain under [scripts/datasets_ingest](scripts/datasets_ingest). No code changes.
- **Files Touched**: [docs/CODE_DESCRIPTION.md](docs/CODE_DESCRIPTION.md), [docs/AGENTIC_HISTORY.md](docs/AGENTIC_HISTORY.md).
- **Commands**: none (doc-only update).
- **Validation**: Link sanity check; verified missing paths are no longer referenced.

## Phase 26 — Distillation scaffold (runtime untouched)
- **Intent**: Stand up training-only teacher→student distillation with real RGB GeoTIFF patches while leaving the QGIS runtime unchanged.
- **Summary**: Added GeoTIFF patch loader (`training/data/geo_patch_dataset.py`), teacher interfaces with Dinov2 adapter + fake fallback (`training/teachers/*`), student embedding CNN (`training/models/student_cnn.py`), distillation/clustering losses (`training/losses_distill.py`), and CLI trainer (`training/train_distill.py`). Updated docs (CODE_DESCRIPTION, TRAINING_PIPELINE, TRAINING_BASELINE, MODEL_HISTORY, DATASETS). No runtime code or exports were modified.
- **Files Touched**: training/data/geo_patch_dataset.py, training/teachers/teacher_base.py, training/teachers/dinov2.py, training/models/student_cnn.py, training/losses_distill.py, training/train_distill.py, training/tests/test_student_embed.py, training/tests/test_teacher_fallback.py, docs/CODE_DESCRIPTION.md, docs/training/TRAINING_PIPELINE.md, docs/training/TRAINING_BASELINE.md, docs/training/MODEL_HISTORY.md, docs/dataset/DATASETS.md.
- **Commands**: none yet (pending full pytest/compileall after code additions).
- **Validation**: Deferred; ensure tests remain QGIS-free and that defaults still use synthetic path.
