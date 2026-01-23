<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Agentic Required Inputs

Purpose: minimal list of files to upload when generating agentic prompt iterations; keep paths stable.

## Upload checklist (exact paths)
- docs/CODE_DESCRIPTION.md
- docs/CODESTATE.md
- docs/AGENTIC_HISTORY.md
- docs/AGENTIC_REQUIRED.md
- docs/plugin/ARCHITECTURE.md
- docs/plugin/MODEL_NOTES.md
- docs/plugin/RUNTIME_STATUS.md
- docs/training/TRAINING_PIPELINE.md
- docs/training/MODEL_HISTORY.md
- docs/training/TRAINING_BASELINE.md
- docs/dataset/DATASETS.md
- docs/dataset/HEADERS.md
- training/README.md

## Regen rules
- If any checklist file is missing or moved, regenerate docs/CODE_DESCRIPTION.md to reflect new locations.
- Append-only for docs/AGENTIC_HISTORY.md; never rewrite prior phases.
- Update this file whenever doc paths or required prompt inputs change.

## Zip archive command
```bash
cd docs && zip agent.zip CODE_DESCRIPTION.md CODESTATE.md AGENTIC_HISTORY.md AGENTIC_REQUIRED.md plugin/ARCHITECTURE.md plugin/MODEL_NOTES.md plugin/RUNTIME_STATUS.md training/TRAINING_PIPELINE.md training/MODEL_HISTORY.md training/TRAINING_BASELINE.md dataset/DATASETS.md dataset/HEADERS.md ../training/README.md
```

## Notes
The following paths referenced in older docs do NOT exist in the repo:
- `model/` directory (runtime_numpy.py, runtime_backend.py, runtime_torch.py, README.md)
- `scripts/datasets_ingest/` (ingestion scaffold CLI and providers)
- `configs/datasets/` (legacy NAIP/3DEP example configs)

Tests that previously referenced these paths have been removed.

## Test Coverage (current)
- **Plugin tests** (tests/): 91 passing, 4 skipped (QGIS/GPU gated)
- **Training tests** (training/tests/): 82 passing
- **Total**: 173 tests

## Security & Linting (QGIS recommended)
Plugin runtime is validated with bandit, detect-secrets, and flake8 per QGIS guidelines.

```bash
# All three checks must pass before release
.venv/bin/bandit -r __init__.py segmenter.py segmenter_dialog.py funcs.py qgis_funcs.py \
  dependency_manager.py raster_utils.py autoencoder_utils.py runtime/ --format txt
.venv/bin/detect-secrets scan *.py runtime/*.py
.venv/bin/flake8 *.py runtime/ --max-line-length=120 --ignore=E501,W503
```

See [plugin/RUNTIME_STATUS.md](plugin/RUNTIME_STATUS.md#security--linting-qgis-recommended) for detailed expectations.