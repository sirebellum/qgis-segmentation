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
