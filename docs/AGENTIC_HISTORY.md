<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# AGENTIC_HISTORY_RESET
- Purpose: Phase 0 snapshot for downstream agentic prompts; derived from [docs/CODESTATE.md](CODESTATE.md). Existing history is retained separately; this file anchors the reset.

## Phase 0 — Current State
- Scope: QGIS plugin + TorchScript legacy runtime (CNN + scikit-learn K-Means) with GDAL rendering; next-gen numpy runtime artifacts and selector exist but are not wired. Stated invariant aims for numpy-only runtime, creating a gap vs current code.
- Training: RGB-only PyTorch scaffold with synthetic default and shard/patch loaders; variable-K monolithic model (stride-4 encoder, soft k-means head, optional refiner); two-view unsupervised losses; exports numpy artifacts (`model.npz`, `meta.json`, `metrics.json`) via weight rename; distillation path adds teacher/student embeddings but no runtime adoption yet.
- Dataset ingestion: Header + shard builder is authoritative (deterministic splits, metrics_train holdout, uncompressed GeoTIFF shards, index.jsonl). Ingestion scaffold under scripts/datasets_ingest is stub-only (manifest planning, no IO/network). GDAL helpers exist for geo math; processed roots have offline fallbacks.
- Runtime contracts: Plugin enforces GDAL 3-band `.tif/.tiff` layers; tiles CNN (192–768px), optional blur, latent KNN; renders GeoTIFF via qgis_funcs. dependency_manager vendors torch/numpy/sklearn unless skipped.
- Tests: Offline pytest suite covers runtime numpy export/tiling, alignment helpers, shard loader/metrics, distillation fallback, ingestion stub; QGIS smoke is skip-gated. GPU/torch-specific paths have limited coverage; QGIS imports remain risky for default runs.
- Risks/Gaps: Runtime numpy-only invariant vs TorchScript reality; manifest->shard ingestion not implemented; learned refine not supported in runtime exports; teacher/student artifacts lack runtime contract.

## Phase Template
- Intent:
- Summary:
- Files Touched:
- Commands:
- Validation:
- Risks/Notes:

## Phase 1 — Runtime export fix + GeoPatch tests (2026-01-17)
- Summary: `smoke_export_runtime` now emits numpy runtime artifacts (meta.json/model.npz) alongside TorchScript outputs; added GeoTIFF patch dataset tests covering RGB/target loading and rotation validation.
- Files Touched: [training/export.py](training/export.py), [training/tests/test_geo_patch_dataset.py](training/tests/test_geo_patch_dataset.py), [docs/AGENTIC_REQUIRED.md](docs/AGENTIC_REQUIRED.md).
- Commands: `.venv/bin/python -m pytest -q`; `.venv/bin/python -m compileall .`.
- Validation: default pytest green (97 passed, 5 skipped); compileall succeeded; existing TorchScript `TracerWarning` and rasterio NotGeoreferenced warnings remain.
- Risks/Notes: `docs/CODESTATE.md` now present; TracerWarnings stem from TorchScript tracing guards and are unchanged.

## Phase 2 — Multires student distillation (2026-01-18)
- Summary: Added three-slice StudentCNN with coarse→mid→fine fusion and per-slice VGG-style 3×3 deep blocks; per-slice distill/clustering/TV losses now merged via EMA-normalized geometric mean with optional cross-resolution consistency; new distillation knobs + CLI overrides; docs/tests updated; runtime untouched.
- Files Touched: training/config.py, training/models/student_cnn.py, training/losses_distill.py, training/train_distill.py, training/tests/test_student_embed.py, training/tests/test_multires_losses.py, docs/training/TRAINING_PIPELINE.md, docs/training/TRAINING_BASELINE.md, docs/training/MODEL_HISTORY.md, training/README.md, docs/CODE_DESCRIPTION.md, docs/AGENTIC_HISTORY.md.
- Commands: `.venv/bin/python -m pytest training/tests/test_student_embed.py training/tests/test_multires_losses.py`
- Validation: targeted pytest passed (6 tests).
- Risks/Notes: New multires path increases training complexity; monitor param budget (<10M) and stability of EMA-normalized merge; runtime remains legacy TorchScript path.

## Phase 3 — Doc reconciliation for removed NAIP AWS scripts (2026-01-19)
- Summary: Updated CODE_DESCRIPTION to mark historical NAIP AWS helper scripts as removed and avoid referencing missing modules; no runtime or training code changes.
- Files Touched: docs/CODE_DESCRIPTION.md, docs/AGENTIC_HISTORY.md.
- Commands: (docs-only) none.
- Validation: manual doc review; required file checklist intact.

## Phase 4 — Patch-size single-scale distillation docs/tests (2026-01-19)
- Summary: Aligned docs to the new single-scale (stride 4) student trained across patch sizes (256/512/1024) with per-scale EMA normalization and geometric-mean metric. Cleaned TRAINING_BASELINE to remove multires duplicates, marked multires phase historical in CODE_DESCRIPTION, updated CODESTATE and MODEL_HISTORY, and recorded the change. Runtime untouched.
- Files Touched: docs/training/TRAINING_BASELINE.md, docs/CODE_DESCRIPTION.md, docs/CODESTATE.md, docs/training/MODEL_HISTORY.md.
- Commands: `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m compileall .`; `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m pytest`.
- Validation: compileall succeeded; pytest passed (103 passed, 5 skipped) with existing GDAL/Tracer warnings unchanged.
- Risks/Notes: Multires path remains available but not active; stride-4 student still TorchScript legacy in runtime.

## Phase 5 — SLIC fallback + shard compatibility (2026-01-19)
- Summary: Made shard loader backward compatible by synthesizing zero-label SLIC maps when absent; shard builder now falls back to a single-label SLIC npz when OpenCV/ximgproc is unavailable (CI-safe) and retains preview plumbing. All pytest suites pass with the fallback path.
- Files Touched: training/data/sharded_tif_dataset.py, training/datasets/build_shards.py, docs/AGENTIC_HISTORY.md.
- Commands: `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m pytest` (full suite green, 106 passed, 5 skipped).
- Validation: Full pytest green; existing rasterio/Tracer warnings persist.
- Risks/Notes: Fallback SLIC is degenerate (single segment) and should be replaced with real SLIC when cv2 is available; previews still depend on cv2 being present.
