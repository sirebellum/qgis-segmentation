<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# CODE_DESCRIPTION

- Purpose: concise registry of modules and their current phase of stewardship (replaces CODE_SCRIPTURE.md).

## Runtime (current — legacy TorchScript + K-Means)
- segmenter.py / segmenter_dialog.py / segmenter_dialog_base.ui: QGIS UI + task dispatch for the legacy runtime; validates 3-band GDAL GeoTIFFs and segment count; builds heuristic/blur overrides and queues `legacy_cnn_segmentation` or `legacy_kmeans_segmentation` via `QgsTask`.
- funcs.py: numerical engine for legacy flows (materialize/tiling/stitching, cancellation/status helpers); TorchScript CNN path via `predict_cnn`, scikit-learn K-Means via `predict_kmeans`, optional torch-accelerated cluster assignment, and optional blur smoothing. Next-gen helpers remain but are not invoked by the plugin.
- models/: packaged TorchScript CNN weights (`model_4/8/16.pth`) loaded by `segmenter.load_model`. Next-gen numpy artifacts under model/best exist for future work but are unused.
- qgis_funcs.py: GDAL render to GeoTIFF + layer registration.
- dependency_manager.py / raster_utils.py: dependency bootstrap (torch, NumPy, scikit-learn) with env overrides and array utilities; adaptive batching uses static defaults (no profiling shim).
- model/runtime_backend.py / model/runtime_numpy.py / model/runtime_torch.py: future runtime selector + numpy/torch implementations for the new model type — present but not wired into the current plugin; update deferred until the new model is trained.
- model/README.md: artifact contract for the deferred next-gen runtime.

## Training (Phase 3, scaffolding)
- training/config.py, config_loader.py: dataclass configs + python-loader overrides.
- training/data/: synthetic dataset placeholder with paired-view augmentations.
- training/augmentations.py: deterministic geometry (90° rotations, H/V flips) + photometric (noise/contrast/saturation) transforms synchronized across RGB/SLIC/targets for shard, geo, and synthetic loaders.
- training/models/: encoder stride/4, soft k-means head, refinement lanes, monolithic wrapper (later iterations removed elevation FiLM).
- training/losses.py, metrics.py: unsupervised objectives + proxy metrics.
- training/utils/: warp/gradients/resample/seed helpers.
- training/train.py, eval.py: CLI runners (synthetic-ready, eager PyTorch).
- training/tests/: pytest coverage (shapes, losses, synthetic smoke).

## Dataset prep (headers + shards)
- training/datasets/header_schema.py: YAML schema + validator for dataset headers (modalities, pairing, splits, sharding, validation rules).
- training/datasets/generate_headers.py: scanner to emit headers from extracted data; ms_buildings auto-detection covers sat/map patterns and label values.
- training/datasets/build_shards.py: monolithic shard builder producing uncompressed GeoTIFF shards with `index.jsonl`, deterministic split assignment, and summary manifest.
- training/datasets/metrics.py: IoU helper that ignores labels ≤ threshold; exercised via training/datasets/tests.

## Training (Phase 4, hardening)
- Elevation mask plumbing (since removed in Phase 13) and grad-accumulation tweaks.
- Fast smoothing fixed to depthwise kernel for arbitrary K; optional grad accumulation exposed in CLI/train loop.
- Synthetic-first default retained; real raster IO still stubbed.
- Docs aligned: ARCHITECTURE.md, MODEL_NOTES.md, training/README.md, training/MODEL_HISTORY.md.

## Training (Phase 5, NAIP/3DEP scaffolding — removed)
- Historical NAIP/3DEP ingestion tooling (mosaicking, DEM alignment, manifest loaders) was removed from the repository; only legacy configs remain for reference. Current ingestion is provided by offline stubs under [scripts/datasets_ingest](../scripts/datasets_ingest).
- configs/datasets/naip_3dep_example.yaml persists as a historical sample but is not wired into the active loaders.

## Training (Phase 7, NAIP-on-AWS refactor — removed)
- Historical NAIP-on-AWS ingestion helpers (prepare scripts and provider modules) were deleted; no runtime code references them. The AWS-specific configs remain for archival reference only.
- configs/datasets/naip_aws_3dep_example.yaml persists as a historical sample but is not consumed by the current loaders.

## Training (Phase 6, export to runtime)
- training/export.py: converts MonolithicSegmenter checkpoints to numpy artifacts (`model.npz`, `meta.json`, `metrics.json`).
- training/train.py: tracks best loss and auto-exports to `model/best` and `training/best_model` (configurable; can disable with `--no-export`).
- tests/test_funcs_runtime.py: runtime engine regression tests (tiling/segmentation shapes, blur, adaptive batching/performance guardrails).

## Docs (Phase 8)
- All supporting docs live under [docs/plugin](../plugin), [docs/training](../training), and [docs/dataset](.). History is tracked at [docs/AGENTIC_HISTORY.md](../AGENTIC_HISTORY.md); required prompt inputs listed in [docs/AGENTIC_REQUIRED.md](../AGENTIC_REQUIRED.md).

## Tests (Phase 8)
- [tests/test_alignment_invariants.py](../../tests/test_alignment_invariants.py): geotransform/bounds helpers (`derive_utm_epsg`, pixel sizes, tolerance).
- [tests/test_export_to_numpy_runtime.py](../../tests/test_export_to_numpy_runtime.py): export-to-runtime pipeline and probability normalization.
- [tests/test_numpy_runtime_tiling.py](../../tests/test_numpy_runtime_tiling.py): numpy runtime tiling/blending smoke via stub runtime.
- [tests/test_runtime_smoke_export.py](../../tests/test_runtime_smoke_export.py): deterministic synthetic-trainer smoke export and runtime contract/meta validation.
- [tests/test_qgis_runtime_smoke.py](../../tests/test_qgis_runtime_smoke.py): optional QGIS import smoke (skips unless `QGIS_TESTS=1`).

## Notes
- Current runtime uses the legacy TorchScript CNN + K-Means path; next-gen numpy runtime artifacts/selectors remain present but are deferred until the new model is trained.
- Real raster IO in training is stubbed behind optional rasterio/gdal; synthetic paths remain the CI-safe default.

## Ops (Phase 10)
- Validation-only pass: python -m compileall . (pass) and ./.venv/bin/python -m pytest -q (44 passed, 1 skipped); no runtime changes.
- System python lacks pytest; use the repo venv for default test invocation to keep offline checks green.

## Ops (Historical Phase 11 — NAIP AWS dry-run hardening; removed)
- Historical branch added a `prepare_naip_aws_3dep_dataset.py` dry-run path and provider helpers; those files are no longer present after the reset and are kept here only as lineage notes.
- No current module implements this behavior; NAIP AWS ingest remains out of scope for the restored tree.

## Ops (Historical Phase 12 — NAIP AWS real-run fallback; removed)
- Historical sample-data and DEM override flags for `prepare_naip_aws_3dep_dataset.py` were removed with the same reset; the script no longer exists in the repository.
- Documentation is retained for provenance only and should not be treated as an active module description.

## Training (Phase 13 — RGB-only reset)
- Intent: remove elevation/DEM inputs while dataset ingestion is rewritten; keep runtime numpy path intact.
- Summary: reintroduced `training/data/` with RGB-only `SyntheticDataset` + `UnsupervisedRasterDataset`; dropped NAIP/3DEP manifest tests; model/loss/export/runtime paths no longer mention elevation metadata.
- Validation: pending full rerun of compileall/pytest after dataset tooling refresh; synthetic smoke remains default.

## Training (Phase 25 — Teacher→Student distillation scaffold)
- Added GeoTIFF patch loader (`training/data/geo_patch_dataset.py`) for RGB-only 512x512 windows with optional aligned targets and config fields `data.raster_paths`/`data.target_paths`.
- Introduced training-only teacher adapters (`training/teachers/teacher_base.py`, `training/teachers/dinov2.py`) with a fake fallback for offline tests; teachers stay out of runtime code.
- Added lightweight student embedding model (`training/models/student_cnn.py`) and distillation/clustering losses (`training/losses_distill.py`), plus CLI trainer (`training/train_distill.py`). Runtime remains the legacy TorchScript path until the student is production-ready.

## Ops (Phase 14 — history reset + ingest scaffold)
- Re-wrote [docs/AGENTIC_HISTORY.md](AGENTIC_HISTORY.md) to phase 0 for current code state.
- Seeded dataset ingestion scaffold under [scripts/datasets_ingest](../scripts/datasets_ingest) (config, interfaces, manifest validation, provider stubs, CLI); scaffold is offline-only and performs no network/GDAL work.
- Documented dataset rewrite status in [docs/dataset/DATASETS.md](dataset/DATASETS.md); added stub tests [tests/test_datasets_ingest_stub.py](../tests/test_datasets_ingest_stub.py) to keep coverage deterministic and QGIS-free.

## Ops (Phase 15 — runtime snapshot doc)
- Added [docs/plugin/RUNTIME_STATUS.md](plugin/RUNTIME_STATUS.md) as a token-efficient snapshot of the QGIS runtime (UI → task → numpy engine → render) with contracts, config points, and known gaps. Snapshot has since been updated to reflect the restored legacy TorchScript path.
- Linked [docs/plugin/ARCHITECTURE.md](plugin/ARCHITECTURE.md) to the new snapshot and recorded the iteration in [docs/AGENTIC_HISTORY.md](AGENTIC_HISTORY.md).

## Docs (Phase 16 — training baseline snapshot)
- Added [docs/training/TRAINING_BASELINE.md](training/TRAINING_BASELINE.md) summarizing current RGB-only training flow, synthetic data default, ingestion scaffold status, export/runtime contract, and offline tests.
- Corrected dataset doc reference to [docs/dataset/DATASETS.md](dataset/DATASETS.md).

## Ops (Phase 17 — runtime invariants)
- Enforced hard 3-band `.tif/.tiff` validation in `segmenter.py` and `funcs.py` with user-facing reasons.
- (Historical) Removed torch/prefetch helpers during the prior numpy-only phase; the current code has been restored to the legacy TorchScript path.
- Shipped stub runtime artifacts under `model/best` for packaging (`pb_tool.cfg` `extra_dirs`); document GitHub artifact + Git LFS expectation.
- Added runtime snapshot update [docs/plugin/RUNTIME_STATUS.md](plugin/RUNTIME_STATUS.md) and offline regression tests guarding dependency specs, torch-free runtime, model artifacts, and 3-band validation.

## Runtime (Historical Phase 18 — torch backend selector; superseded by restore)
- Historical branch added [model/runtime_backend.py](../model/runtime_backend.py) to prefer torch and fall back to numpy; the restored plugin does **not** load runtimes via this selector. Modules remain for future work only.
- Historical env overrides (`SEGMENTER_RUNTIME_BACKEND`, `SEGMENTER_DEVICE`, `SEGMENTER_ENABLE_TORCH`, etc.) are not honored by the restored legacy flow; current dependency bootstrap simply installs torch/NumPy/scikit-learn unless skipped.
- Backend-selection tests referenced here apply to the historical numpy-first branch, not the current code.

## Ops (Historical Phase 19 — import/package hardening; superseded by restore)
- Earlier numpy-only branch introduced QGIS/PyQt stubs and relative-import guards. The restored legacy TorchScript path again imports QGIS/PyQt directly and is not importable outside QGIS.
- Historical import/package tests remain in the tree but do not reflect the current runtime constraints.

## Ops (Historical Phase 20 — torch bootstrap knobs; superseded by restore)
- Historical branch added extra pip knobs (`SEGMENTER_TORCH_EXTRA_INDEX_URL`, `SEGMENTER_TORCH_PIP_ARGS`, `SEGMENTER_ENABLE_TORCH`). The restored bootstrap honors only `SEGMENTER_SKIP_AUTO_INSTALL`, `SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`, `SEGMENTER_PYTHON`, and `SEGMENTER_PIP_EXECUTABLE`.
- GPU wheel selection is determined by index URL; there is no runtime toggle in the restored code.

## Docs (Phase 21 — restored runtime doc sync)
- Reconciled runtime docs (ARCHITECTURE, MODEL_NOTES, RUNTIME_STATUS) to match the restored legacy TorchScript/K-Means plugin path; marked next-gen numpy/torch runtime as deferred until the new model is trained.
- Updated CODE_DESCRIPTION and AGENTIC_HISTORY accordingly; retained historical phase notes with clarification where the runtime differed.
- Scope limited to documentation; runtime code unchanged.

## Docs (Phase 22 — doc reconciliation to restored legacy runtime)
- Re-read current legacy runtime code (TorchScript CNN + scikit-learn K-Means) and aligned docs to match; code now overrides historical numpy-path notes where they conflicted.
- Clarified historical phases (backend selector, import stubs, torch bootstrap knobs) as superseded; future runtime update remains deferred until the new model is trained.
- Files updated: docs/plugin/ARCHITECTURE.md, docs/plugin/MODEL_NOTES.md, docs/plugin/RUNTIME_STATUS.md, docs/CODE_DESCRIPTION.md, docs/AGENTIC_HISTORY.md.
- Validation: `python -m compileall .` and `python -m pytest -q` (default QGIS-free suite); results captured in AGENTIC_HISTORY.

## Training (Phase 24 — shard ingestion wiring)
- Added shard-backed iterable loader [training/data/sharded_tif_dataset.py](training/data/sharded_tif_dataset.py) with worker-partitioned shard streaming and optional per-worker LRU caching (`data.cache_mode=lru`, `data.cache_max_items`). Loader perf knobs (`data.num_workers`, `data.prefetch_factor`, `data.persistent_workers`, `data.pin_memory`) are honored across synthetic and shard paths.
- [training/train.py](training/train.py) now honors `data.source=shards`, builds shard DataLoaders, and runs IoU metrics on `metrics_train`/`val` splits using [training/datasets/metrics.py](training/datasets/metrics.py) (labels `<=0` ignored). [training/eval.py](training/eval.py) accepts shard overrides via CLI (`--data-source shards --dataset-id ... --split ...`).
- Tests: [training/tests/test_sharded_dataset_loader.py](training/tests/test_sharded_dataset_loader.py) (worker partitioning, caching determinism) and [training/tests/test_metrics_iou_ignore_zero.py](training/tests/test_metrics_iou_ignore_zero.py).

## Ops (Phase 26 — runtime smoke export hardening)
- `smoke_export_runtime` now writes numpy runtime artifacts (meta/model) alongside the TorchScript export so runtime loaders and backend selectors find `meta.json`/`model.npz` in smoke outputs ([training/export.py](training/export.py)).
- Added GeoTIFF patch dataset coverage for RGB/target loading and rotation validation ([training/tests/test_geo_patch_dataset.py](training/tests/test_geo_patch_dataset.py)).
- Validation: `.venv/bin/python -m pytest -q` (97 passed, 5 skipped) and `.venv/bin/python -m compileall .`.

## Training (Historical Phase 27 — multires student distillation)
- `StudentEmbeddingNet` emitted three resolution slices (stride 16/8/4) with coarse→mid→fine fusion and per-slice VGG-style 3×3 deep stacks (GroupNorm default). Config knobs lived under `student.*` and `distill.*` for dims/depths/norm, clustering iters, EMA merge decay/eps, and loss weights.
- `train_distill.py` applied per-slice feature+affinity distill, soft k-means pseudo labels, edge-aware TV, cross-resolution consistency, and a scale-neutral EMA-normalized geometric-mean merge; CLI overrides included `--disable-multires`, `--student-dims`, `--student-depths`, `--student-norm`, `--consistency-weight`, `--ema-decay`, `--cluster-iters`.
- Tests covered multires shape/stride + deep-block invariants ([training/tests/test_student_embed.py](../training/tests/test_student_embed.py)) and merge/consistency behavior ([training/tests/test_multires_losses.py](../training/tests/test_multires_losses.py)). Superseded by Phase 28.

## Training (Phase 28 — patch-size single-scale distillation)
- StudentEmbeddingNet is now a single stride-4 embedding path shared across patch sizes (default 256/512/1024). Patch size is sampled per step (uniform) or summed across scales; losses per patch size (feature/affinity distill, clustering, edge-aware TV) are normalized by per-scale EMA, and a virtual geometric-mean metric enforces scale neutrality. CLI keeps `--patch-sizes`, `--patch-size-sampling`, `--multi-scale-mode`, `--ema-decay`, and student knobs (embed dim/depth/norm/dropout).
- `train_distill.py` builds per-size loaders once, samples the size each step, and logs normalized per-scale losses plus the virtual geometric mean. Runtime remains unchanged (legacy TorchScript path).
- Tests exercise stride-4 outputs for 256/512/1024 inputs ([training/tests/test_student_embed.py](../training/tests/test_student_embed.py)) and per-scale EMA/geometric-mean merge invariants ([training/tests/test_multires_losses.py](../training/tests/test_multires_losses.py)).
