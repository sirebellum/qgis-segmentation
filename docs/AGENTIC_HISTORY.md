<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Agentic History

New series beginning at the current repository state. Phase numbering restarts here; Phase 0 captures the present code snapshot and invariants.

## Phase 0 — Current State (Reset)
- **Runtime**: QGIS plugin runs a single numpy-only path. `Segmenter.predict` in [segmenter.py](segmenter.py#L620-L697) queues `predict_nextgen_numpy` from [funcs.py](funcs.py#L32-L88) with tiling/stitching and cancellation. Rendering writes GeoTIFF via [qgis_funcs.py](qgis_funcs.py#L7-L45). Dependency bootstrap is NumPy-only in [dependency_manager.py](dependency_manager.py#L1-L204); perf tuner is a no-op shim ([perf_tuner.py](perf_tuner.py#L1-L41)). UI stripped to layer picker + segment count ([segmenter_dialog_base.ui](segmenter_dialog_base.ui)).
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
