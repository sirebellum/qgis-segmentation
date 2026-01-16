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
- **Intent**: Resolve QGIS plugin load failures (ModuleNotFoundError for `funcs`) by making intra-plugin imports package-safe and adding regression coverage.
- **Summary**: Switched runtime imports to explicit relative form, added lightweight QGIS/PyQt stubs so [segmenter.py](segmenter.py) can import without bindings, and guarded `Segmenter` instantiation when QGIS is absent. Bootstrapped the `segmenter` package name in [conftest.py](conftest.py) to mirror the deployed plugin folder, added offline import/regression tests ([tests/test_plugin_imports.py](tests/test_plugin_imports.py)) and updated existing tests to use package paths. Optional QGIS smoke now honors `RUN_QGIS_TESTS=1` (legacy `QGIS_TESTS=1` still works) and covers `classFactory` discovery.
- **Files Touched**: [segmenter.py](segmenter.py), [conftest.py](conftest.py), [func_test.py](func_test.py), [tests/test_runtime_invariants.py](tests/test_runtime_invariants.py), [tests/test_runtime_backend_selection.py](tests/test_runtime_backend_selection.py), [tests/test_runtime_torch_gpu.py](tests/test_runtime_torch_gpu.py), [tests/test_numpy_runtime_tiling.py](tests/test_numpy_runtime_tiling.py), [tests/test_qgis_runtime_smoke.py](tests/test_qgis_runtime_smoke.py), [tests/test_plugin_imports.py](tests/test_plugin_imports.py), [docs/CODE_DESCRIPTION.md](docs/CODE_DESCRIPTION.md).
- **Commands**: Not run (offline changes only).
- **Validation**: Not run (CI/default pytest expected to cover new import tests once executed).
- **Risks/Notes**: QGIS-dependent behavior remains stubbed in non-QGIS environments; runtime execution still requires actual QGIS bindings and GDAL. Ensure the deployed plugin folder remains named `segmenter` so relative imports resolve.

## Phase 7 — Torch bootstrap knobs

## Phase 8 — Sync runtime docs to restored original runtime
- **Intent**: Align runtime-facing docs with the restored legacy TorchScript CNN + scikit-learn K-Means code and clarify that the new model runtime integration is deferred.
- **Summary**: Reconciled ARCHITECTURE, MODEL_NOTES, RUNTIME_STATUS, CODE_DESCRIPTION, and AGENTIC_HISTORY against the actual plugin code; marked historical runtime selector/import/torch-bootstrap phases as superseded; reiterated that next-gen numpy runtime is deferred until training finishes.
- **Files Touched**: [docs/plugin/ARCHITECTURE.md](plugin/ARCHITECTURE.md), [docs/plugin/MODEL_NOTES.md](plugin/MODEL_NOTES.md), [docs/plugin/RUNTIME_STATUS.md](plugin/RUNTIME_STATUS.md), [docs/CODE_DESCRIPTION.md](docs/CODE_DESCRIPTION.md), [docs/AGENTIC_HISTORY.md](AGENTIC_HISTORY.md).
- **Commands**: `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m compileall .` (pass); `/Users/josh/gits/qgis-segmentation/.venv/bin/python -m pytest -q` (fails during collection: `ModuleNotFoundError: No module named 'qgis'` from runtime tests importing segmenter.py). No remediation in this doc-only pass.
- **Validation**: Compileall succeeded. Pytest requires QGIS bindings for runtime tests; documented failure and left unresolved per scope.
- **Risks/Notes**: Runtime update for the new model type remains deferred until training is complete. Default test suite currently depends on QGIS for runtime modules; consider gating or stubbing if QGIS-free runs are required in future iterations.
