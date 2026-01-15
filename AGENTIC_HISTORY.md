<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# AGENTIC_HISTORY

## Phase 0 — Baseline (2026-01-15)
- Intent: Establish Phase 0 baseline and generate internal docs of record from repo inspection.
- Summary:
  - Created ARCHITECTURE.md and MODEL_NOTES.md from verified code structure (segmenter UI, funcs engine, dependency/perf tooling, rendering path).
  - Added AGENTIC_HISTORY.md to track agent work; noted license duality (existing BSD-3-Clause, new docs carry MIT SPDX per request).
- Files Touched:
  - Added: ARCHITECTURE.md, MODEL_NOTES.md, AGENTIC_HISTORY.md.
  - Unmodified: existing source, LICENSE retained as BSD-3-Clause.
- Commands:
  - pytest ./func_test.py (pre-existing run, exit code 130; no new commands executed in this phase).
- Validation:
  - Confirmed absence of prior ARCHITECTURE/MODEL_NOTES/AGENTIC_HISTORY before creation.
  - Cross-checked module paths referenced in docs exist: segmenter.py, funcs.py, qgis_funcs.py, dependency_manager.py, perf_tuner.py, autoencoder_utils.py, raster_utils.py, models/ files.
- Risks/Notes:
  - License mismatch: repo LICENSE is BSD-3-Clause while new docs use MIT SPDX headers; reconcile licensing direction in a future phase.

## Phase 1 — Training Docs + License Align (2026-01-15)
- Intent: Align doc headers to BSD-3-Clause and add training contract/history scaffolding per unsupervised plan.
- Summary:
  - Switched ARCHITECTURE.md, MODEL_NOTES.md, AGENTIC_HISTORY.md headers to BSD-3-Clause for consistency with repo LICENSE.
  - Added training/README.md with unlabeled data layout and model export contract notes; added training/MODEL_HISTORY.md for recording losses/eval choices.
- Files Touched:
  - Modified: ARCHITECTURE.md, MODEL_NOTES.md, AGENTIC_HISTORY.md.
  - Added: training/README.md, training/MODEL_HISTORY.md.
- Commands:
  - pytest ./func_test.py (pre-existing run, exit code 0; no new commands run this phase).
- Validation:
  - Verified new training docs reference existing modules and exported contract expected by `predict_cnn`.
  - Confirmed training/ path created with BSD headers.
- Risks/Notes:
  - Unsupervised training details still TBD; MODEL_HISTORY.md is a placeholder to be populated with experiments.

## Phase 2 — Dead-Code Cleanup (2026-01-15)
- Intent: Remove unused segmentation branches (chunked helpers/texture autoencoder) while preserving legacy K-Means/CNN flows and sync contributor docs.
- Summary:
  - Deleted texture autoencoder module and chunked segmentation/blur helpers; kept only legacy kmeans/cnn paths.
  - Trimmed tests and packaging config to drop deleted modules; refreshed contributor guidance and model notes accordingly.
- Files Touched:
  - Modified: funcs.py, func_test.py, pb_tool.cfg, ARCHITECTURE.md, MODEL_NOTES.md, .github/copilot-instructions.md.
  - Removed: autoencoder_utils.py.
- Commands:
  - /Users/josh/gits/qgis-segmentation/.venv/bin/python -m compileall /Users/josh/gits/qgis-segmentation (pass).
- Validation:
  - compileall succeeded; no runtime tests run this phase.
- Risks/Notes:
  - Latent doc references to autoencoder/chunked paths were scrubbed; future updates should add back only if new entrypoints are wired through Segmenter.

## Phase 3 — Training Scaffolding (2026-01-15)
- Intent: Add eager-PyTorch unsupervised training scaffold (variable-K model with optional elevation) without touching runtime inference.
- Summary:
  - Introduced training package: configs, data/augmentations (paired views, elevation dropout), synthetic dataset, monolithic model (encoder stride/4, FiLM elevation gate, soft k-means head, fast/learned refinement lanes), unsupervised losses/metrics, utilities, CLI train/eval runners.
  - Added pytest coverage for shapes/losses/synthetic smoke training.
  - Updated docs (training README, MODEL_HISTORY, MODEL_NOTES, ARCHITECTURE) and added CODE_DESCRIPTION for new modules.
- Files Touched:
  - Added: training/ (config.py, config_loader.py, train.py, eval.py, losses.py, metrics.py, data/, models/, utils/, tests/).
  - Modified: training/README.md, training/MODEL_HISTORY.md, MODEL_NOTES.md, ARCHITECTURE.md, AGENTIC_HISTORY.md, CODE_DESCRIPTION.md.
- Commands:
  - Not run in this phase: compileall/pytest (instructions provided in docs for future execution).
- Validation:
  - Static reasoning on shapes/contracts; synthetic-focused tests included but not executed here.
- Risks/Notes:
  - Real raster IO remains a stub (rasterio optional); runtime integration deferred; TorchScript export intentionally omitted.

## Phase 4 — Training Hardening (2026-01-15)
- Intent: Harden the unsupervised training scaffold with elevation masks, gradient accumulation, and smoothing correctness while keeping runtime untouched.
- Summary:
  - Added per-sample elevation masks through collate → FiLM gate; kept dropout behavior and mask-aware smoothness inputs.
  - Fixed fast smoothing depthwise kernel for arbitrary K; wired optional grad accumulation into CLI loop and logs.
  - Kept synthetic-first default in train/eval; refreshed docs (ARCHITECTURE, MODEL_NOTES, training/README, training/MODEL_HISTORY) and added CODE_SCRIPTURE registry.
- Files Touched:
  - Modified: training/models/{model.py,elevation_inject.py,refine.py}, training/train.py, training/eval.py, training/README.md, training/MODEL_HISTORY.md, ARCHITECTURE.md, MODEL_NOTES.md.
  - Added: CODE_SCRIPTURE.md.
- Commands:
  - None executed this phase (reasoned changes only; tests not run here).
- Validation:
  - Static inspection of tensor shapes, mask broadcasting, grad-accum loop; existing pytest suite expected to cover shapes/losses/smoke.
- Risks/Notes:
  - Real raster IO still stubbed; dataloader masks assume consistent spatial dims. Synthetic-first stance maintained until real data backend is implemented.

## Phase 5 — NAIP/3DEP Data Scaffolding (2026-01-15)
- Intent: Add NAIP RGB + 3DEP DEM dataset prep CLI, manifest loader wiring, and docs without touching runtime inference.
- Summary:
  - Added GDAL/TNM helpers and `scripts/data/prepare_naip_3dep_dataset.py` to query TNM, download/mosaic/clip, reproject to UTM, align DEM to NAIP with -tap, tile 512x512/stride 128, and emit manifest.jsonl.
  - Introduced `training/data/naip_3dep_dataset.py` manifest loader with per-tile DEM standardization and hooked manifest loading into `training.train` (YAML config support in config_loader; PyYAML optional but listed in requirements).
  - Added sample config `configs/datasets/naip_3dep_example.yaml`, docs (`docs/DATASETS.md`, `docs/TRAINING_PIPELINE.md`), and loader unit test scaffold.
- Files Touched:
  - Added: scripts/data/_gdal_utils.py, scripts/data/_usgs_tnm_provider.py, scripts/data/prepare_naip_3dep_dataset.py, training/data/naip_3dep_dataset.py, configs/datasets/naip_3dep_example.yaml, docs/DATASETS.md, docs/TRAINING_PIPELINE.md, training/tests/test_naip_3dep_manifest_loader.py.
  - Modified: training/config.py, training/config_loader.py, training/train.py, training/README.md, requirements.txt, CODE_DESCRIPTION.md.
- Commands:
  - python -m compileall . (pass)
  - python -m pytest training/tests/test_naip_3dep_manifest_loader.py -q (failed: No module named pytest in system Python)
- Validation:
  - compileall succeeded; manifest loader test not executed due to missing pytest binary in PATH/system interpreter.
- Risks/Notes:
  - Requires GDAL CLI availability and network access to TNM; DEM tier fallback applied when 1 m unavailable. PyYAML needed for YAML configs; pytest missing in system interpreter (install to run tests).

## Phase 6 — NextGen Numpy Runtime Wiring (2026-01-15)
- Intent: Begin runtime integration of the variable-K model via a numpy-only path and auto-export best checkpoints to `model/best` without breaking legacy K-Means/CNN flows.
- Summary:
  - Added numpy runtime (`model/runtime_numpy.py`, `model/__init__.py`) plus smoke script and packaging updates; plugin exposes "Next-Gen (Numpy)" option and dispatches through `predict_nextgen_numpy` with graceful missing-artifact messaging.
  - Training now tracks best loss and exports numpy artifacts via `training/export.py`, writing to `model/best` and `training/best_model`; new artifact contract documented in `model/README.md`.
  - Added dummy runtime test in `func_test.py`; ignored bulky training artifacts in `.gitignore`; updated pb_tool to ship runtime module.
- Files Touched:
  - Added: model/__init__.py, model/runtime_numpy.py, model/README.md, scripts/smoke_runtime_nextgen.py, training/export.py.
  - Modified: segmenter.py, funcs.py, training/train.py, func_test.py, pb_tool.cfg, .gitignore, CODE_DESCRIPTION.md, MODEL_NOTES.md, ARCHITECTURE.md.
- Commands:
  - Not run in this phase (pending user confirmation and environment constraints).
- Validation:
  - Static reasoning only; added unit smoke for numpy runtime (not executed here). Legacy tests not re-run.
- Risks/Notes:
  - Numpy runtime may be slow on large rasters; learned refine/elevation unsupported in this phase. Legacy TorchScript weights still required for CNN mode; next-gen artifacts must be exported via training CLI before use.
