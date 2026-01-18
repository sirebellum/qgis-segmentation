<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# CODESTATE

## Overview
- Repo provides a QGIS plugin for map segmentation plus an isolated PyTorch training/export stack and dataset prep scaffold. Runtime code currently uses legacy TorchScript CNN + scikit-learn K-Means paths despite the intended numpy-only invariant; next-gen numpy runtime modules exist but are not wired into the plugin. Training focuses on RGB-only unsupervised segmentation with shard ingestion; dataset ingestion rewrite is mid-scaffold.

## Path Map
- Plugin runtime: [segmenter.py](../segmenter.py), [segmenter_dialog.py](../segmenter_dialog.py), [segmenter_dialog_base.ui](../segmenter_dialog_base.ui), [qgis_funcs.py](../qgis_funcs.py), [dependency_manager.py](../dependency_manager.py), [funcs.py](../funcs.py).
- Model artifacts/runtimes: [models](../models) (TorchScript), [model/runtime_numpy.py](../model/runtime_numpy.py), [model/runtime_torch.py](../model/runtime_torch.py), [model/runtime_backend.py](../model/runtime_backend.py), [model/README.md](../model/README.md).
- Training core: [training/train.py](../training/train.py), [training/eval.py](../training/eval.py), [training/export.py](../training/export.py), configs in [training/config.py](../training/config.py) + [training/config_loader.py](../training/config_loader.py), data loaders in [training/data](../training/data), models in [training/models](../training/models), losses/metrics in [training/losses.py](../training/losses.py) and [training/metrics.py](../training/metrics.py), distillation in [training/train_distill.py](../training/train_distill.py) + [training/losses_distill.py](../training/losses_distill.py) + [training/teachers](../training/teachers).
- Dataset prep: header/shard tooling in [training/datasets](../training/datasets) (generate headers, build shards, schema, IoU metrics); ingestion scaffold in [scripts/datasets_ingest](../scripts/datasets_ingest).
- Utilities: GDAL helpers [scripts/data/_gdal_utils.py](../scripts/data/_gdal_utils.py); warp/seed/grad utils under [training/utils](../training/utils).
- Tests: repo tests under [tests](../tests) and [training/tests](../training/tests) plus [training/datasets/tests](../training/datasets/tests).

## Entrypoints and CLIs
- Training: `python -m training.train` (shards or synthetic) [training/train.py](../training/train.py); `python -m training.eval` for metrics proxy [training/eval.py](../training/eval.py); distillation via `python -m training.train_distill` (teacher→student) [training/train_distill.py](../training/train_distill.py).
- Export: `python -m training.export --smoke-out ...` for deterministic numpy runtime artifacts [training/export.py](../training/export.py).
- Dataset prep: `python -m training.datasets.generate_headers` builds header YAMLs [training/datasets/generate_headers.py](../training/datasets/generate_headers.py); `python -m training.datasets.build_shards` creates processed shards [training/datasets/build_shards.py](../training/datasets/build_shards.py).
- Ingestion scaffold: `python -m scripts.datasets_ingest.cli --list-providers` etc. for stub manifest planning [scripts/datasets_ingest/cli.py](../scripts/datasets_ingest/cli.py).

## Training: Architecture and Flow
- Model: [training/models/model.py](../training/models/model.py) `MonolithicSegmenter` = stride-4 encoder ([training/models/backbone.py](../training/models/backbone.py)), soft k-means head ([training/models/soft_cluster.py](../training/models/soft_cluster.py)), optional learned refiner ([training/models/refine.py](../training/models/refine.py)), fast smoothing lane default.
- Forward contract: rgb [B,3,H,W], k∈[2,max_k], returns probs/logits/embeddings/prototypes/logits_latent; learned refine pads channels when k<max_k; fast smoothing uses depthwise box filter.
- Knobs: random k, cluster_iters, smooth_iters, downsample, smoothing_lane sampled per step (`sample_knobs`) with ranges from [training/config.py](../training/config.py).
- Data: synthetic RGB tensors via [training/data/synthetic.py](../training/data/synthetic.py) wrapped into two augmented views with flips/rotations/noise and identity warp grids [training/data/dataset.py](../training/data/dataset.py). Shard-backed iterable loader streams GeoTIFF shards with per-worker partitioning/caching and optional targets [training/data/sharded_tif_dataset.py](../training/data/sharded_tif_dataset.py). GeoTIFF patch dataset samples random 512px windows (optionally with aligned targets) [training/data/geo_patch_dataset.py](../training/data/geo_patch_dataset.py).
- Losses: two-view consistency (symmetric KL on warped probs), entropy min/max, edge-aware smoothness [training/losses.py](../training/losses.py). Proxy metrics for eval include utilization/speckle/boundary/self-consistency [training/metrics.py](../training/metrics.py). Distillation losses cover feature/affinity distill, clustering sharpness/balance, edge-aware TV [training/losses_distill.py](../training/losses_distill.py).
- Trainer: [training/train.py](../training/train.py) builds DataLoader (synthetic or shard source), runs AMP-enabled loop with grad accumulation, logs JSON/TensorBoard, optional eval on metrics/val splits via masked IoU [training/datasets/metrics.py](../training/datasets/metrics.py), exports best numpy artifacts if enabled. Eval runner mirrors shard/synthetic paths [training/eval.py](../training/eval.py).
- Export contract: [training/export.py](../training/export.py) renames checkpoint weights to runtime schema (stem/block1/block2/seed_proj), writes `model.npz`, `meta.json`, `metrics.json`, sets `supports_learned_refine=false`, stride=4, max_k/embed_dim/temperature defaults. Smoke export path produces deterministic CPU artifact.

## Dataset Ingestion: Architecture and Flow
- Headers: schema parser/validator [training/datasets/header_schema.py](../training/datasets/header_schema.py) defines modalities, pairing, splits (ratios/manifests/metrics holdout), sharding, validation (IoU ignore threshold). Generators for ms_buildings/whu_building/openearth/inria infer stats and emit YAML [training/datasets/generate_headers.py](../training/datasets/generate_headers.py).
- Sharding: [training/datasets/build_shards.py](../training/datasets/build_shards.py) loads headers, pairs inputs/targets by stem or manifest, assigns splits deterministically (optional metrics_train carve-out for labeled fraction), copies uncompressed GeoTIFFs into `processed/<split>/<dataset>/shard-xxxxx/{inputs,targets}` with `index.jsonl` entries and summary metadata. Fallback extracted/processed roots under training/datasets/data support offline smoke.
- Loader contract: shard index fields `input`, optional `target`, `item_id`, `split`, `has_target`; loader remaps target labels >0 to contiguous IDs and masks labels<=ignore threshold during IoU metrics.
- Ingestion scaffold: [scripts/datasets_ingest](../scripts/datasets_ingest) defines `BaseProvider`/`IngestConfig` and stub providers (placeholder, NAIP AWS stub) returning manifest entries only—no IO/network. CLI validates manifest structure and lists providers; serves as rewrite placeholder.

## Runtime (Plugin): Interfaces and Contracts
- UI/task flow in [segmenter.py](../segmenter.py): validates GDAL layer (.tif/.tiff, provider gdal, 3 bands), parses segment count/resolution, builds blur/heuristic overrides, dispatches QgsTask to `legacy_cnn_segmentation` (TorchScript) or `legacy_kmeans_segmentation` [funcs.py](../funcs.py). Device preference CUDA→MPS→CPU; sliders tune tile size (192–768px), blur kernel, sampling scale.
- Numerical engine [funcs.py](../funcs.py): GDAL materialization, tiling, CNN inference via TorchScript model_provider, scikit-learn KMeans with optional torch GPU assignment, latent KNN refine, legacy blur smoothing, cancellation token checks. Rendering writes GeoTIFF via [qgis_funcs.py](../qgis_funcs.py) preserving CRS/extent.
- Dependency bootstrap: [dependency_manager.py](../dependency_manager.py) installs torch/numpy/scikit-learn into vendor unless `SEGMENTER_SKIP_AUTO_INSTALL`; torch index/spec overridable. Despite numpy-only intent, plugin imports torch on load for the current legacy path.

## Artifacts and File Layouts
- Legacy runtime weights: TorchScript models `models/model_4/8/16.pth` loaded by UI resolution choice.
- Next-gen runtime artifacts (numpy) in [model/best](../model/best) (`model.npz`, `meta.json`, `metrics.json`) produced by training/export; not consumed by plugin.
- Processed shards: `training/datasets/processed/<split>/<dataset>/shard-xxxxx/{inputs,targets}/`, `index.jsonl`, summary JSON under `metadata/` [training/datasets/build_shards.py](../training/datasets/build_shards.py).
- Headers under [training/datasets/headers](../training/datasets/headers); fallback extracted/processed sample data under [training/datasets/data](../training/datasets/data).
- Logs/exports: training writes TensorBoard to `checkpoint_dir/tb`; best numpy export to `model/best` plus mirror `training/best_model` if configured.

## Dependency/Tooling Surface
- Plugin deps: torch, numpy, scikit-learn auto-installed via [dependency_manager.py](../dependency_manager.py); GDAL required for raster IO/render.
- Training deps: PyTorch (CUDA/MPS aware), tensorboard, numpy; rasterio for shard/patch loading; yaml for header generation (optional), json/rasterio in build_shards; tests assume rasterio and torch available.
- Ingestion scaffold is pure-Python, no network/GDAL invoked; GDAL helpers in [scripts/data/_gdal_utils.py](../scripts/data/_gdal_utils.py) rely on osgeo.

## Determinism and Reproducibility Hooks
- Global seeding via [training/utils/seed.py](../training/utils/seed.py) used in train/eval/export; shard split assignment seeded (header.splits.seed or CLI override) [training/datasets/build_shards.py](../training/datasets/build_shards.py).
- Smoke export fixes steps/K/patch size for deterministic numpy artifacts [training/export.py](../training/export.py).
- Stochastic elements: data aug flips/rotations/noise, per-step knob sampling, DataLoader worker partitioning, torch nondeterminism on GPU; shard caching order depends on worker sharding; teacher loading may vary (DINO download fallback to FakeTeacher).

## Current Test Inventory
- Runtime/export: [tests/test_export_to_numpy_runtime.py](../tests/test_export_to_numpy_runtime.py), [tests/test_numpy_runtime_tiling.py](../tests/test_numpy_runtime_tiling.py), [tests/test_funcs_runtime.py](../tests/test_funcs_runtime.py), [tests/test_runtime_invariants.py](../tests/test_runtime_invariants.py) (legacy runtime expectations), [tests/test_runtime_backend_selection.py](../tests/test_runtime_backend_selection.py), [tests/test_runtime_torch_gpu.py](../tests/test_runtime_torch_gpu.py), [tests/test_runtime_smoke_export.py](../tests/test_runtime_smoke_export.py), [tests/test_plugin_imports.py](../tests/test_plugin_imports.py), optional QGIS smoke [tests/test_qgis_runtime_smoke.py](../tests/test_qgis_runtime_smoke.py) (skip unless QGIS available).
- Training: [training/tests/test_model_shapes.py](../training/tests/test_model_shapes.py), [training/tests/test_losses.py](../training/tests/test_losses.py), [training/tests/test_smoke_train.py](../training/tests/test_smoke_train.py), [training/tests/test_synthetic.py](../training/tests/test_synthetic.py), distillation tests [training/tests/test_student_embed.py](../training/tests/test_student_embed.py), [training/tests/test_teacher_fallback.py](../training/tests/test_teacher_fallback.py), shard loader/caching/IoU tests [training/tests/test_sharded_dataset_loader.py](../training/tests/test_sharded_dataset_loader.py), [training/tests/test_metrics_iou_ignore_zero.py](../training/tests/test_metrics_iou_ignore_zero.py).
- Dataset headers/shards: [training/datasets/tests/test_pipeline.py](../training/datasets/tests/test_pipeline.py), [training/datasets/tests/test_dataset_generators.py](../training/datasets/tests/test_dataset_generators.py).
- Ingestion scaffold stub: [tests/test_datasets_ingest_stub.py](../tests/test_datasets_ingest_stub.py).
- Alignment/GDAL helpers: [tests/test_alignment_invariants.py](../tests/test_alignment_invariants.py).

## Coverage Risk Register
- Plugin runtime depends on QGIS/GDAL/PyQt; default pytest suite must avoid importing QGIS (some tests optional). Legacy TorchScript path conflicts with stated numpy-only invariant; risk of regressions when switching runtimes.
- GDAL/rasterio IO in shard/patch loaders and build_shards requires sample data; hard to exercise without fixtures. Large file copies in shard builder not unit-tested beyond dry-run.
- Distillation path depends on optional DINOv2 download; fallback mitigates but coverage limited to fake teacher.
- Latent KNN, GPU paths, and torch-accelerated KMeans assignments in [funcs.py](../funcs.py) are complex and lightly covered.
- Manifest/ingestion scaffold is stub-only; real network/COG flows untested by design.

## Testing Targets (Functional Checklist)
- Training: knob sampling bounds; two-view loss outputs; AMP/grad-accum correctness; shard loader caching/partitioning; IoU masking; distillation losses and teacher fallback; smoke export deterministic artifacts.
- Dataset ingestion: header validation errors; generator outputs per dataset; split assignment (metrics_train carve-out, ratio handling); shard index fields and label remap; dry-run vs overwrite behavior; fallback extracted/processed roots.
- Export/contract validation: state_dict rename coverage; meta.json defaults; runtime_numpy predict_labels vs exported artifacts; supports_learned_refine flag adherence.
- Runtime core (non-QGIS): funcs tiling/stitching, blur, latent KNN refinement, adaptive settings; dependency_manager env knobs; runtime backend selector behavior for torch/numpy paths.
- Optional QGIS integration (skip-gated): plugin layer validation, task cancellation/progress parsing, render_raster CRS/extent preservation, UI wiring.

## Spec Gaps / Ambiguities
- Declared invariant says plugin runtime should be numpy-only, but current code/doc flow uses TorchScript CNN + scikit-learn; future migration path and timing are unclear.
- `DataConfig.manifest_path` is unused; ingestion scaffold does not realize manifests into shards yet.
- Distillation path produces training checkpoints only; export/runtime expectations for student model are not defined.
