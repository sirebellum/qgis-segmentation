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

## Phase 7 — NAIP on AWS + 3DEP Alignment (2026-01-15)
- Intent: Refactor Option 3 dataset prep to source NAIP RGB from AWS COGs (requester-pays) with USGS 3DEP DEM, keep warp-to-grid alignment, 512x512 tiling, and manifest outputs for training-only use.
- Summary:
  - Added AWS provider + index VRT builder (`scripts/data/_naip_aws_provider.py`) and USGS 3DEP product selector/downloader (`scripts/data/_usgs_3dep_provider.py`); extended GDAL helpers with grid spec/warp/VRT and env propagation for requester-pays.
  - New CLI `scripts/data/prepare_naip_aws_3dep_dataset.py` builds NAIP VRT from AWS index, warps NAIP/DEM to shared grid (-tap), tiles 512/stride 128, skips high-nodata chips, and records DEM tier/gsd + NAIP metadata in manifest.
  - Loader now accepts extended manifest fields (nodata_fraction, NAIP URLs/year/state, DEM native/resampled/target gsd) and exposes an AWS alias module; added sample config `configs/datasets/naip_aws_3dep_example.yaml`.
  - Docs updated: DATASETS (NAIP-on-AWS flow), TRAINING_PIPELINE (new script/alias/config), CODE_DESCRIPTION (Phase 7 entry).
- Files Touched:
  - Added: scripts/data/_naip_aws_provider.py, scripts/data/_usgs_3dep_provider.py, scripts/data/prepare_naip_aws_3dep_dataset.py, training/data/naip_aws_3dep_dataset.py, configs/datasets/naip_aws_3dep_example.yaml.
  - Modified: scripts/data/_gdal_utils.py, training/data/naip_3dep_dataset.py, docs/DATASETS.md, docs/TRAINING_PIPELINE.md, CODE_DESCRIPTION.md.
- Commands:
  - python -m compileall . (pass; ran in venv with GDAL available on PATH assumed).
- Validation:
  - compileall succeeded; no dataset prep or pytest runs executed this phase.
- Risks/Notes:
  - Requires GDAL CLI and network access; AWS buckets are requester-pays (set AWS_REQUEST_PAYER). 3DEP tier ladder falls back to coarser DEM when 1 m unavailable; manifest preserves gsd/tier for downstream QC. Tile vetting uses nodata fraction threshold only; imagery quality/capture dates not filtered.

## Phase 8 — Test Hardening + Docs Restructure (2026-01-15)
- Intent: add offline/optional coverage across manifest/schema, numpy runtime tiling/export, and record doc relocations plus agentic input checklist.
- Summary:
  - Added pytest modules for manifest parsing, geotransform invariants, numpy runtime export/forward, stub tiling, and optional QGIS import smoke (skips unless `QGIS_TESTS=1`).
  - Documented doc relocation into `docs/plugin`, `docs/training`, `docs/dataset`; added docs/AGENTIC_REQUIRED.md with upload checklist; refreshed CODE_DESCRIPTION and training README links.
- Files Touched:
  - Added: tests/test_manifest_schema.py, tests/test_alignment_invariants.py, tests/test_export_to_numpy_runtime.py, tests/test_numpy_runtime_tiling.py, tests/test_qgis_runtime_smoke.py, docs/AGENTIC_REQUIRED.md.
  - Modified: docs/dataset/CODE_DESCRIPTION.md, training/README.md, docs/AGENTIC_HISTORY.md (this entry).
- Commands:
  - Not run in this phase (static updates; tests not executed here).
- Validation:
  - Static reasoning only; tests expected to be deterministic/offline; QGIS smoke gated by env var.
- Risks/Notes:
  - QGIS coverage remains optional; dataset prep still depends on GDAL/network outside test stubs. Ensure AGENTIC_REQUIRED stays aligned with future doc moves.

## Phase 9 — Numpy-Only Runtime Simplification (2026-01-15)
- Intent: remove legacy K-Means/CNN TorchScript paths, torch dependency, and profiling hooks so the plugin runs exclusively on the monolithic next-gen numpy runtime.
- Summary:
  - Simplified plugin runtime and UI to a single next-gen path; removed model/resolution/heuristic controls and torch-based code paths.
  - Trimmed dependency bootstrap to NumPy-only and converted perf_tuner to a no-op shim.
  - Updated docs (architecture, model notes, code description, README) to reflect numpy-only behavior; refreshed func_test coverage accordingly.
- Files Touched:
  - Modified: segmenter.py, funcs.py, dependency_manager.py, perf_tuner.py, segmenter_dialog_base.ui, func_test.py, docs/plugin/ARCHITECTURE.md, docs/plugin/MODEL_NOTES.md, docs/dataset/CODE_DESCRIPTION.md, README.md, docs/AGENTIC_HISTORY.md (this entry).
- Commands:
  - None executed this phase (code/doc edits only).
- Validation:
  - Static reasoning; tests not run in this phase.
- Risks/Notes:
  - Ensure `model/best` artifacts exist before running segmentation; legacy TorchScript CNNs are no longer consumed by the plugin UI/runtime.

## Phase 10 — Offline Stabilization Check (2026-01-15)
- Intent: verify numpy-only runtime invariants and confirm offline tests/compileall pass without code changes.
- Summary:
  - Ran compileall across the repo (venv python) — success.
  - Default `python -m pytest -q` failed on system python (pytest missing); reran with repo venv and all tests passed (44 passed, 1 skipped, ~5s).
  - No source/runtime changes made; validation only.
- Files Touched:
  - Modified: docs/AGENTIC_HISTORY.md (this entry), docs/CODE_DESCRIPTION.md (phase note update).
- Commands:
  - python -m compileall .
  - python -m pytest -q (system python; fails: No module named pytest)
  - ./.venv/bin/python -m pytest -q
- Validation:
  - compileall: pass.
  - pytest (venv): pass; torch absent from runtime path per existing tests.
- Risks/Notes:
  - System python lacks pytest; use repo venv for default test invocation. No runtime/state changes performed.

## Phase 11 — NAIP AWS Dry-Run Hardening (2026-01-16)
- Intent: Make the NAIP AWS + 3DEP dataset prep dry-run succeed offline and add deterministic tests.
- Summary:
  - Added stub NAIP index + DEM selection for `--dry-run`, skipping GDAL checks and network fetches.
  - Enhanced NAIP provider with GeoJSON-first parsing, bbox overlap filter, and cached/stub handling.
  - Introduced offline dry-run tests and fixture to validate stub index path.
- Files Touched:
  - Modified: scripts/data/prepare_naip_aws_3dep_dataset.py, scripts/data/_naip_aws_provider.py, docs/dataset/DATASETS.md, docs/CODE_DESCRIPTION.md.
  - Added: tests/test_prepare_naip_aws_3dep_dry_run.py, tests/fixtures/naip_index_min.geojson, docs/dataset/CODE_DESCRIPTION.md.
- Commands:
  - /Users/josh/gits/qgis-segmentation/.venv/bin/python scripts/data/prepare_naip_aws_3dep_dataset.py --output-dir /tmp/naipaws3dep --dry-run --seed 123
  - /Users/josh/gits/qgis-segmentation/.venv/bin/python -m compileall .
  - /Users/josh/gits/qgis-segmentation/.venv/bin/python -m pytest -q
- Validation:
  - Dry-run command exits 0 with stub index/DEM logs; compileall + pytest (46 passed, 1 skipped) succeeded in repo venv.
- Risks/Notes:
  - Full (non-dry-run) path still requires GDAL CLI + network; stub index is dry-run only.

## Phase 12 — NAIP AWS Real-Run Fallback (2026-01-16)
- Intent: Fix real-run failures (NAIP index 403, TNM DNS resolution) by enabling a minimal real download path without AWS/TNM credentials and tightening requester-pays handling.
- Summary:
  - Added `--sample-data` mode to `prepare_naip_aws_3dep_dataset.py` that downloads tiny public GitHub rasters (rgbsmall.tif/byte.tif) and exercises the full warp/tile/manifest pipeline; clamped patch/stride for small sources.
  - Introduced DEM override flags (`--dem-url`, `--dem-id`, `--dem-native-gsd`) and Requester Pays headers + actionable 403 error for NAIP index fetches.
  - Documented minimal real download recipe in DATASETS.md and recorded the new ops phase in CODE_DESCRIPTION.
- Files Touched:
  - Modified: scripts/data/prepare_naip_aws_3dep_dataset.py, scripts/data/_naip_aws_provider.py, docs/dataset/DATASETS.md, docs/CODE_DESCRIPTION.md.
  - Modified tests: tests/test_prepare_naip_aws_3dep_dry_run.py (new overrides/header coverage).
- Commands:
  - /Users/josh/gits/qgis-segmentation/.venv/bin/python scripts/data/prepare_naip_aws_3dep_dataset.py --output-dir /tmp/naipaws3dep-real --cities chicago_il --aoi-size-m 600 --patch-size 256 --stride 256 --prefer-1m-dem 0 --target-gsd 10 --use-https --seed 7 (failed: NAIP index 403 Requester Pays).
  - curl -I https://naip-visualization.s3.amazonaws.com/index/naip_footprint.gpkg (403 AccessDenied without credentials).
  - curl -I https://tnmaccess.usgs.gov/api/v1/products (DNS resolution failed).
  - /Users/josh/gits/qgis-segmentation/.venv/bin/python -m compileall .
  - /Users/josh/gits/qgis-segmentation/.venv/bin/python -m pytest -q
  - /Users/josh/gits/qgis-segmentation/.venv/bin/python scripts/data/prepare_naip_aws_3dep_dataset.py --output-dir /tmp/naipaws3dep-sample --sample-data --patch-size 32 --stride 32 --target-gsd 30 --use-https --seed 7
- Validation:
  - compileall: pass; pytest: pass (including new override/header tests).
  - Sample real download exits 0; manifest and tiles written under /tmp/naipaws3dep-sample/data/naip_aws_3dep/.
- Risks/Notes:
  - True AWS/TNM runs still require credentials/network; Requester Pays now surfaces actionable errors. Sample mode is for smoke validation, not production data quality.

## Phase 13 — Elevation Removal & RGB-Only Training (2026-01-16)
- Intent: strip elevation/DEM dependencies from training and runtime contracts while dataset ingestion is being rewritten.
- Summary:
  - Added RGB-only data stack (`training/data/{dataset.py,synthetic.py}`) and retired NAIP/DEM manifest tests.
  - Simplified `MonolithicSegmenter`, losses, and train/eval to drop elevation inputs; removed elevation FiLM module; export/runtime metadata no longer advertises elevation support.
  - Updated docs (ARCHITECTURE, MODEL_NOTES, TRAINING_PIPELINE, CODE_DESCRIPTION) to reflect RGB-only path.
- Files Touched: training/config.py, training/train.py, training/eval.py, training/losses.py, training/models/model.py, training/models/elevation_inject.py (removed), training/data/* (new), tests/train*, tests/test_manifest_schema.py (removed), model/runtime_numpy.py, training/export.py, docs/{plugin/ARCHITECTURE.md,plugin/MODEL_NOTES.md,training/TRAINING_PIPELINE.md,CODE_DESCRIPTION.md,AGENTIC_HISTORY.md}.
- Commands: not yet rerun in this phase (compileall/pytest recommended post-dataset rewrite).
- Validation: pending; synthetic-only defaults expected to keep offline pytest green once rerun.
- Risks/Notes: Manifest/DEM ingestion temporarily disabled; downstream datasets will need reintroduction once new ingestion lands.
